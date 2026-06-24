#!/usr/bin/env python3
"""
LCAO-to-Wannier90 Multi-Stage Workflow Script

This script implements the Wannier90 workflow with three stages:

Stage 1: Generate .win file from CRYSTAL/LCAO output
Stage 2: Generate .eig, .amn, .mmn files using .nnkp neighbor information
Stage 3: Symmetrize wannier90_hr.dat using crystal symmetry (Reynolds operator)
Stage 4: Plot LCAO band structure with PDWF projectability coloring

Usage:
    # Stage 1: Create .win file
    python lcao_to_wannier90.py --stage 1 --input material.out --seedname material

    # Then run: wannier90.x -pp material

    # Stage 2: Create data files using .nnkp
    python lcao_to_wannier90.py --stage 2 --input material.out --seedname material

    # Then run: wannier90.x material

    # Stage 3: Symmetrize the tight-binding Hamiltonian
    python lcao_to_wannier90.py --stage 3 --input material.out --seedname material

    # Stage 4: Plot band structure with projectability
    python lcao_to_wannier90.py --stage 4 --input material.out --seedname material

For help:
    python lcao_to_wannier90.py --help
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

from lcao_wannier import (
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    parse_atomic_basis_info,
    create_spin_block_matrices,
    create_nonsoc_full_matrices,
    prepare_real_space_matrices,
    Wannier90Engine,
    suggest_optimal_window,
    analyze_band_window,
    parse_atoms_from_crystal_output,
    estimate_fermi_energy,
)
from lcao_wannier.projectability import select_bands_by_projectability, smart_select_bands
from lcao_wannier.parser import parse_orbital_types
from lcao_wannier.basis_parser import parse_basis_shells, get_atom_list
from lcao_wannier.valence_config import (
    build_target_mask, compute_num_wann, summarize_config,
)
from lcao_wannier.lcao_pdwf import (
    compute_lowdin_projectability, classify_bands, determine_windows,
    check_frozen_interlopers, check_band_count, print_pdwf_summary,
    ClassificationParams,
)

# Threshold above which --method direct warns the user
LARGE_BASIS_THRESHOLD = 200

# Maximum allowed discrepancy between CRYSTAL-reported Fermi and
# HOMO estimated from (eigenvalues, num_electrons). Beyond this, the
# reported Fermi is deemed inconsistent with the printed H(R) matrices
# (e.g. SPINLOCK+level-shifter quirk in CRYSTAL23 SOC outputs) and we
# fall back to the electron-count estimate with a prominent warning.
FERMI_FRAME_TOLERANCE_EV = 1.0


def _sanity_check_fermi_energy(engine, params, args, stage_name=""):
    """
    Compare the parsed Fermi energy against the HOMO estimated from
    eigenvalues + electron count. Detects CRYSTAL SPINLOCK/level-shifter
    frame mismatches where the Fermi and H(R) matrices end up in different
    energy references.

    If `--fermi-energy` was passed explicitly, trusts that and returns.
    If the parsed Fermi disagrees with the electron-count estimate by
    more than FERMI_FRAME_TOLERANCE_EV, replaces engine.e_fermi with the
    estimate and prints a loud warning.

    Must be called AFTER engine.solve_all_kpoints().
    """
    # User override always wins — they've made a conscious choice.
    if getattr(args, 'fermi_energy', None) is not None:
        return

    # Can't sanity-check without a parsed Fermi and electron count.
    if params.fermi_energy is None or params.num_electrons is None:
        return

    # Compute HOMO-based estimate from our eigenvalues.
    # Spin degeneracy: SOC systems have spinor bands (1 electron/band);
    # non-SOC systems have spatial bands (2 electrons/band in the
    # non-spin-polarized case treated by our solver).
    has_soc = getattr(params, 'has_soc', False) or getattr(engine, 'has_soc', False)
    spin_deg = 1 if has_soc else 2
    try:
        estimated = estimate_fermi_energy(
            engine.eigenvalues_list,
            num_electrons=params.num_electrons,
            method='electron_count',
            spin_degeneracy=spin_deg,
        )
    except Exception as exc:
        print(f"  ⚠ Could not run Fermi sanity check: {exc}")
        return

    discrepancy = abs(params.fermi_energy - estimated)
    if discrepancy <= FERMI_FRAME_TOLERANCE_EV:
        # Parsed and estimated agree — nothing to do.
        return

    # Discrepancy is large. Almost always means the CRYSTAL output had
    # SPINLOCK active or a level-shifter-altered Fermi (e.g. two-component
    # SOC runs with LOCKING - FERMI ENERGY ALTERED BY LEVEL SHIFTER).
    stage_tag = f" ({stage_name})" if stage_name else ""
    print()
    print("!" * 78)
    print(f"  ⚠  FERMI ENERGY FRAME MISMATCH DETECTED{stage_tag}")
    print("!" * 78)
    print(f"  Parsed from CRYSTAL output : {params.fermi_energy:+.4f} eV")
    print(f"  Estimated from band filling: {estimated:+.4f} eV "
          f"(N_electrons={params.num_electrons})")
    print(f"  Discrepancy                : {discrepancy:.4f} eV "
          f"(tolerance {FERMI_FRAME_TOLERANCE_EV} eV)")
    print()
    print("  The parsed Fermi and the printed H(R) matrices appear to live")
    print("  in different energy reference frames. Common causes in CRYSTAL23:")
    print("    - SPINLOCK active ('LOCKING - FERMI ENERGY ALTERED BY LEVEL")
    print("       SHIFTER' in the output)")
    print("    - 'EIGENVALUE LEVEL SHIFTING OF X HARTREE' applied but not")
    print("       removed before the Fermi-energy header was written")
    print("    - Two-component SOC SCF with an internal Fermi-bias field")
    print()
    print(f"  → Falling back to the electron-count estimate "
          f"({estimated:+.4f} eV).")
    print(f"  → To override this, re-run with: "
          f"--fermi-energy <your_value_in_eV>")
    print("!" * 78)
    print()

    # Apply the fallback to both params (used later in the stage) and engine.
    params.fermi_energy = estimated
    engine.e_fermi = estimated


def _apply_method_projectability(engine, args, has_soc=False):
    """Apply Method 1: Smart projectability-based band selection."""
    print("\nProjectability-based band selection...")
    print("-" * 80)

    # Check for user override of num_wann
    user_num_wann = getattr(args, 'num_wann', None)

    # Use smart selector if eigenvalues and Fermi energy are available
    if engine.eigenvalues_list and engine.e_fermi is not None:
        result = smart_select_bands(
            engine.eigenvectors_list,
            engine.S_k_list,
            engine.eigenvalues_list,
            e_fermi=engine.e_fermi,
            has_soc=has_soc,
            proj_threshold=args.proj_threshold,
            verbose=True,
        )

        if result.num_wann == 0:
            print("ERROR: Smart selector found no suitable bands!")
            print(f"  Projectability threshold: {args.proj_threshold}")
            print(f"  Try lowering --proj-threshold (e.g., 0.8)")
            sys.exit(1)

        # Use frontier detection for disentanglement
        if user_num_wann is not None:
            # User override: force specific num_wann
            engine.num_wann = user_num_wann
            print(f"\n  User override: num_wann = {user_num_wann}")
        else:
            engine.num_wann = result.recommended_num_wann

        engine.selected_band_indices = result.selected_band_indices

        # Store disentanglement info on engine for win_file generation
        engine._dis_win = result.recommended_dis_win
        engine._dis_froz = result.recommended_dis_froz

        # If num_wann == num selected bands but dis_win suggests more bands,
        # expand selected_band_indices to include the extra bands in the
        # disentanglement window. This ensures num_bands > num_wann.
        num_selected = len(result.selected_band_indices)
        if (engine.num_wann >= num_selected
                and result.recommended_dis_win is not None):
            # Count bands in the recommended dis_win at each k-point
            ef = engine.e_fermi if engine.e_fermi is not None else 0.0
            win_min = ef + result.recommended_dis_win[0]
            win_max = ef + result.recommended_dis_win[1]

            # Find minimum consistent band count across all k-points
            min_bands = None
            for evals in engine.eigenvalues_list:
                count = np.sum((evals >= win_min) & (evals <= win_max))
                if min_bands is None or count < min_bands:
                    min_bands = count

            if min_bands > engine.num_wann:
                # Expand band selection to include extra bands for disentanglement
                # Find the band indices that are in the dis_win at Gamma
                evals_gamma = engine.eigenvalues_list[0]
                in_window = np.where(
                    (evals_gamma >= win_min) & (evals_gamma <= win_max)
                )[0]
                engine.selected_band_indices = in_window[:min_bands]
                engine._num_bands_for_win = min_bands
                print(f"\n  Expanded band selection for disentanglement:")
                print(f"    dis_win captures {min_bands} bands (min across k-points)")
                print(f"    Selected bands: {engine.selected_band_indices}")
            else:
                engine._num_bands_for_win = num_selected
        else:
            engine._num_bands_for_win = num_selected

        if result.recommended_dis_win is not None and engine._num_bands_for_win > engine.num_wann:
            print(f"\nDisentanglement setup:")
            print(f"  num_wann (target):   {engine.num_wann}")
            print(f"  num_bands (total):   {engine._num_bands_for_win}")
            print(f"  dis_win  (rel E_F):  [{result.recommended_dis_win[0]:.4f}, {result.recommended_dis_win[1]:.4f}] eV")
            if result.recommended_dis_froz is not None:
                print(f"  dis_froz (rel E_F):  [{result.recommended_dis_froz[0]:.4f}, {result.recommended_dis_froz[1]:.4f}] eV")
        elif result.recommended_dis_win is not None:
            print(f"\nDisentanglement setup:")
            print(f"  num_wann (frontier): {engine.num_wann}")
            print(f"  num_bands (total):   {engine._num_bands_for_win}")
            print(f"  dis_win  (rel E_F):  [{result.recommended_dis_win[0]:.4f}, {result.recommended_dis_win[1]:.4f}] eV")
            if result.recommended_dis_froz is not None:
                print(f"  dis_froz (rel E_F):  [{result.recommended_dis_froz[0]:.4f}, {result.recommended_dis_froz[1]:.4f}] eV")
        else:
            print(f"\nSelected {engine.num_wann} bands, no disentanglement needed")
        print(f"Quality score: {result.quality_score:.4f}")

    else:
        # Fallback to simple projectability if no eigenvalues/Fermi available
        result = select_bands_by_projectability(
            engine.eigenvectors_list,
            engine.S_k_list,
            threshold=args.proj_threshold,
            verbose=True,
        )

        if result.num_wann == 0:
            print("ERROR: No bands have projectability above threshold!")
            print(f"  Threshold: {args.proj_threshold}")
            print(f"  Try lowering --proj-threshold (e.g., 0.8)")
            sys.exit(1)

        engine.num_wann = result.num_wann
        engine.selected_band_indices = result.selected_band_indices
        print(f"\nSelected {result.num_wann} bands for Wannier functions")

    # Select projection orbitals
    proj_method = getattr(args, 'projection_method', 'weight')
    print(f"\nSelecting projection orbitals (method={proj_method})...")
    print("-" * 80)
    engine.select_projections(verbose=True, method=proj_method)


def _apply_method_pdwf(engine, args, has_soc, lines):
    """Apply LCAO-PDWF method: chemistry-grounded projectability band selection."""
    print("\nLCAO-PDWF band selection...")
    print("-" * 80)

    extended = getattr(args, 'extended', False)
    include_tm_p = getattr(args, 'include_tm_p', False)
    p_high = getattr(args, 'pdwf_p_high', 0.90)
    p_low = getattr(args, 'pdwf_p_low', 0.10)

    # Phase 1: Parse basis shells
    print("  Phase 1: Parsing basis set shells...")
    try:
        from lcao_wannier.parser import parse_calculation_parameters
        params_calc = parse_calculation_parameters(lines)
        num_atoms_hint = params_calc.num_atoms
    except Exception:
        num_atoms_hint = None

    shells, num_ao_spatial = parse_basis_shells(lines, num_atoms=num_atoms_hint)
    atoms = get_atom_list(shells)
    print(f"    Found {len(shells)} shells, {num_ao_spatial} spatial AOs")
    print(f"    Atoms: {atoms}")

    # Phase 2: Build target mask
    print("  Phase 2: Building target orbital mask...")
    target_mask = build_target_mask(
        shells, extended=extended, include_tm_p=include_tm_p,
        has_soc=has_soc, verbose=True,
    )
    num_wann = compute_num_wann(
        atoms, extended=extended, include_tm_p=include_tm_p,
        has_soc=has_soc,
    )
    print(f"\n    Target AOs: {int(np.sum(target_mask))} / {len(target_mask)}")
    print(f"    num_wann: {num_wann}")
    print(summarize_config(atoms, extended=extended,
                           include_tm_p=include_tm_p, has_soc=has_soc))

    # Align target_mask to engine orbital dimension.
    # parse_basis_shells may count more AOs than the H/S matrices contain
    # (e.g. 336 vs 296 spatial AOs in CrI3), causing a mismatch after SOC
    # doubling (672 vs 592). Truncate the spatial part to engine.num_orbitals//2.
    eng_dim = engine.num_orbitals
    mask_len_orig = len(target_mask)
    if mask_len_orig != eng_dim:
        if has_soc:
            spatial_engine = eng_dim // 2
            half_mask = mask_len_orig // 2
            mask_spatial = target_mask[:half_mask][:spatial_engine]
            target_mask = np.concatenate([mask_spatial, mask_spatial])
        else:
            target_mask = target_mask[:eng_dim]
        print(f"    [Note] target_mask trimmed {mask_len_orig}->{eng_dim} "
              f"(basis parser: {num_ao_spatial} spatial AOs, "
              f"engine: {eng_dim // (2 if has_soc else 1)})")

    # Phase 3: Compute Lowdin projectability
    print("\n  Phase 3: Computing Lowdin projectability...")
    eigenvalues = np.array(engine.eigenvalues_list)
    proj = compute_lowdin_projectability(
        engine.eigenvectors_list, engine.S_k_list, target_mask,
    )
    print(f"    Projectability range: [{np.min(proj):.4f}, {np.max(proj):.4f}]")

    # Phase 4: Classify bands
    print("\n  Phase 4: Classifying bands...")
    e_fermi = engine.e_fermi if engine.e_fermi is not None else 0.0
    classification = classify_bands(
        proj, eigenvalues, num_wann,
        ClassificationParams(p_high=p_high, p_low=p_low, e_fermi=e_fermi),
    )

    # Phase 5: Determine windows
    print("\n  Phase 5: Determining disentanglement windows...")
    windows = determine_windows(classification, eigenvalues, e_fermi)

    # Validate
    all_warnings = []
    if windows.dis_froz_min is not None:
        all_warnings += check_frozen_interlopers(
            eigenvalues, classification.frozen_indices,
            classification.excluded_indices,
            windows.dis_froz_min, windows.dis_froz_max,
        )
    if windows.dis_win_min is not None:
        all_warnings += check_band_count(
            eigenvalues, windows.dis_win_min, windows.dis_win_max, num_wann,
        )

    # Print summary
    print_pdwf_summary(classification, windows, eigenvalues,
                       e_fermi, all_warnings)

    # Handle poor projectability gap gracefully
    if classification.gap_quality < 2.0:
        print(f"\n  WARNING: Poor projectability gap (quality = "
              f"{classification.gap_quality:.2f}).")
        if len(classification.frozen_indices) == 0:
            # No frozen bands found — use energy-range-based freezing.
            # Freeze all bands with reasonable projectability below E_F + margin
            froz_margin_above = 3.0  # eV above E_F
            print(f"  No frozen bands from projectability. Using energy-range "
                  f"freezing up to E_F + {froz_margin_above:.1f} eV.")
            # Promote bands with avg_p >= p_low that are mostly below threshold
            avg_e = classification.band_energies
            avg_p = classification.avg_projectability
            for b in range(len(avg_p)):
                if (avg_p[b] >= p_low and
                    avg_e[b] <= e_fermi + froz_margin_above and
                    classification.category[b] != 'excluded'):
                    classification.category[b] = 'frozen'
            classification.frozen_indices = np.where(
                classification.category == 'frozen')[0]
            classification.disent_indices = np.where(
                classification.category == 'disent')[0]
            # Recompute windows with updated classification
            windows = determine_windows(classification, eigenvalues, e_fermi)
            print(f"  Energy-range frozen bands: {len(classification.frozen_indices)}")

    # Apply results to engine
    engine.num_wann = num_wann

    # Selected bands = frozen + disentangle
    all_active = np.union1d(classification.frozen_indices,
                            classification.disent_indices)
    if len(all_active) == 0:
        print("ERROR: No bands classified as frozen or disentangle!")
        sys.exit(1)

    # Ensure band ratio >= 1.5 for adequate disentanglement freedom
    min_ratio = 1.5
    min_bands = max(int(np.ceil(num_wann * min_ratio)), num_wann + 1)
    if len(all_active) < min_bands:
        nb = eigenvalues.shape[1]
        # Find outer window bounds from current active set
        if windows.dis_win_min is not None:
            win_lo = windows.dis_win_min
            win_hi = windows.dis_win_max
        else:
            active_eigs = eigenvalues[:, all_active]
            win_lo = float(np.min(active_eigs)) - 2.0
            win_hi = float(np.max(active_eigs)) + 2.0

        # Expand outer window to include more bands above
        candidates = []
        for b in range(nb):
            if b in set(all_active):
                continue
            band_eigs = eigenvalues[:, b]
            # Include bands that overlap with expanded outer window
            if np.any((band_eigs >= win_lo) & (band_eigs <= win_hi + 10.0)):
                avg_e_b = np.mean(band_eigs)
                candidates.append((avg_e_b, b))

        # Sort candidates by energy (prefer bands near the active set)
        candidates.sort(key=lambda x: x[0])
        for _, b in candidates:
            all_active = np.union1d(all_active, [b])
            if len(all_active) >= min_bands:
                break

        # Update outer window to encompass all active bands
        active_eigs = eigenvalues[:, all_active]
        windows.dis_win_min = float(np.min(active_eigs)) - 2.0
        windows.dis_win_max = float(np.max(active_eigs)) + 2.0

        ratio = len(all_active) / num_wann
        print(f"\n  Expanded band set to {len(all_active)} bands "
              f"(ratio {ratio:.2f}) for adequate disentanglement")

    engine.selected_band_indices = all_active

    # Store disentanglement windows (relative to E_F for win_file)
    if windows.dis_win_min is not None and len(all_active) > num_wann:
        engine._dis_win = (windows.dis_win_min - e_fermi,
                           windows.dis_win_max - e_fermi)
    else:
        engine._dis_win = None

    if windows.dis_froz_min is not None:
        engine._dis_froz = (windows.dis_froz_min - e_fermi,
                            windows.dis_froz_max - e_fermi)
    else:
        engine._dis_froz = None

    engine._num_bands_for_win = len(all_active)

    # Store full target mask for SVD-based PDWF Amn generation
    # (projects onto ALL target AOs, then SVD selects optimal num_wann subspace)
    engine.pdwf_target_mask = target_mask
    engine.selected_orbital_indices = None  # Not used with PDWF Amn

    if windows.dis_win_min is not None and len(all_active) > num_wann:
        ratio = len(all_active) / num_wann
        print(f"\n  Disentanglement enabled:")
        print(f"    num_wann:  {num_wann}")
        print(f"    num_bands: {len(all_active)} (ratio {ratio:.2f})")
        if windows.dis_froz_min is not None:
            print(f"    frozen:    [{windows.dis_froz_min - e_fermi:+.1f}, "
                  f"{windows.dis_froz_max - e_fermi:+.1f}] eV (rel E_F)")
        print(f"    outer:     [{windows.dis_win_min - e_fermi:+.1f}, "
              f"{windows.dis_win_max - e_fermi:+.1f}] eV (rel E_F)")
    else:
        print(f"\n  No disentanglement needed: {num_wann} bands selected")

    # Set iteration counts for proper convergence.
    # Previous values (2000 / 200) regressed MgB2 PDWF from Ω_total = 8.09 Å²
    # to 44.28 Å² — the assumption that "SVD-based PDWF Amn converges fast"
    # is false in practice. Match projectability/symmetry defaults.
    if engine._override_num_iter is None:
        engine._override_num_iter = 10000
    if engine._override_dis_num_iter is None:
        engine._override_dis_num_iter = 5000
    # Guiding centres keeps the initial Wannier centres anchored, critical for
    # PDWF where the SVD-chosen subspace has weak initial localization.
    engine._guiding_centres = True


def _apply_method_direct(engine, args, has_soc, params):
    """Apply Method 2: Direct LCAO orbital mapping."""
    num_basis = engine.num_orbitals  # Already doubled for SOC

    print("\nDirect LCAO orbital mapping...")
    print("-" * 80)
    print(f"  num_basis (total orbitals): {num_basis}")
    if has_soc:
        print(f"  (includes SOC doubling: {params.num_ao} AOs x 2 = {num_basis} spinors)")

    # Safety check for large basis sets
    if num_basis > LARGE_BASIS_THRESHOLD:
        print(f"\n{'!'*70}")
        print(f"  CRITICAL WARNING: Large basis set detected!")
        print(f"  num_basis = {num_basis} exceeds threshold ({LARGE_BASIS_THRESHOLD})")
        print(f"  Wannier90 with {num_basis} Wannier functions will be very slow.")
        print(f"  Recommendation: Try --method projectability first.")
        print(f"{'!'*70}")

        if not args.force:
            try:
                response = input("\n  Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("  Aborted. Use --method projectability or --force to override.")
                    sys.exit(0)
            except EOFError:
                print("  Non-interactive mode: use --force to override. Aborting.")
                sys.exit(1)
        else:
            print("  --force flag set, proceeding...")

    # Set num_wann = num_basis (all orbitals)
    engine.num_wann = num_basis
    engine.selected_band_indices = np.arange(num_basis)

    # Select ALL orbitals as projections
    engine.selected_orbital_indices = np.arange(num_basis)

    # Skip spread minimization
    engine._override_num_iter = 0

    print(f"\n  num_wann = {num_basis} (all LCAO orbitals)")
    print(f"  num_iter will be set to 0 (skip spread minimization)")


def _apply_method_symmetry(engine, args, has_soc, params, lines):
    """Apply Method 3: Symmetry-enforced pre-Wannierization + irrep band selection.

    Steps:
    A) Symmetrize H(R) and S(R) matrices using crystal symmetry
    B) Re-solve eigenvalue problems with symmetrized matrices
    C) Select bands using symmetry-adapted criteria (irrep analysis)
    """
    from lcao_wannier.symmetry import (
        detect_symmetry_operations,
        build_representation_matrices,
        symmetrize_real_space_matrices,
        enforce_hermiticity,
        enforce_time_reversal,
        get_orbital_structure_from_crystal,
    )
    from lcao_wannier.irreps import select_bands_by_symmetry

    tolerance = getattr(args, 'sym_tolerance', 1e-5)

    print("\nSymmetry-enforced method...")
    print("-" * 80)

    # --- Part A: Symmetrize matrices ---

    # 1. Extract crystal structure
    print("  A1. Extracting crystal structure for spglib...")
    atomic_info = parse_atomic_basis_info(lines)
    atom_positions_frac = atomic_info.atom_positions  # fractional
    atom_symbols = atomic_info.atom_symbols

    # Map symbols to atomic numbers for spglib
    from lcao_wannier.symmetry import SymmetryInfo
    _SYMBOL_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    }
    atom_numbers = np.array([_SYMBOL_TO_Z.get(s, 0) for s in atom_symbols])

    lattice_vectors = engine.lattice_vectors
    print(f"  Structure: {len(atom_symbols)} atoms, lattice shape {lattice_vectors.shape}")

    # 2. Detect symmetry operations
    print("  A2. Detecting symmetry operations via spglib...")
    sym_info = detect_symmetry_operations(
        lattice_vectors, atom_positions_frac, atom_numbers, tolerance=tolerance
    )
    print(f"  Space group: {sym_info.space_group}")
    print(f"  Point group: {sym_info.point_group}")
    print(f"  Number of symmetry operations: {sym_info.nsymm}")

    # 3. Get orbital structure
    print("  A3. Parsing orbital structure...")
    orbital_types_dict = parse_orbital_types(lines, has_soc=False, num_atoms=len(atom_symbols))

    if not orbital_types_dict:
        print("  WARNING: Could not parse orbital types from CRYSTAL output.")
        print("  Falling back to projectability method.")
        _apply_method_projectability(engine, args, has_soc=has_soc)
        return

    # Build per-atom orbital shell list
    # Compute per-atom basis function counts from the basis-atom mapping
    per_atom_counts = [int(np.sum(atomic_info.basis_atom_map == a))
                       for a in range(len(atom_symbols))]
    orbital_structure = get_orbital_structure_from_crystal(
        orbital_types_dict, per_atom_counts, len(atom_symbols)
    )
    print(f"  Orbital structure per atom:")
    for i, shells in enumerate(orbital_structure):
        print(f"    Atom {i} ({atom_symbols[i]}): {shells}")

    # 4. Build representation matrices
    print("  A4. Building representation matrices...")
    protmat_list = build_representation_matrices(
        sym_info, orbital_structure, has_soc=has_soc
    )
    print(f"  Representation matrix shape: {protmat_list[0].shape}")

    # 5. Compute orbital offsets and counts
    from lcao_wannier.symmetry import _orbital_dim
    norbs_per_atom = []
    for atom_orbs in orbital_structure:
        n = sum(_orbital_dim(t) for t in atom_orbs)
        norbs_per_atom.append(n)

    orbital_offsets = np.zeros(len(atom_symbols), dtype=int)
    for i in range(1, len(atom_symbols)):
        orbital_offsets[i] = orbital_offsets[i-1] + norbs_per_atom[i-1]
    orbital_counts = np.array(norbs_per_atom, dtype=int)

    # Save original eigenvectors/overlap before symmetrization.
    # Symmetrized eigenstates in centrosymmetric systems have equal weight on
    # both atoms (bonding/antibonding), destroying site-specific character.
    # We use symmetrized eigenvalues for band selection and .eig, but restore
    # the original eigenvectors for AMN/MMN so Wannier90 gets site-localized
    # initial projections.
    original_eigenvectors = list(engine.eigenvectors_list)
    original_S_k = list(engine.S_k_list)

    # 6. Symmetrize real-space matrices
    print("  A5. Symmetrizing H(R) and S(R) matrices...")
    sym_matrices = symmetrize_real_space_matrices(
        engine.real_space_matrices,
        sym_info, protmat_list,
        orbital_offsets, orbital_counts,
        has_soc=has_soc, verbose=True
    )

    # 7. Enforce Hermiticity
    print("  A6. Enforcing Hermiticity: H(R) = [H(R) + H(-R)^dag]/2...")
    sym_matrices = enforce_hermiticity(sym_matrices)

    # 8. Enforce time-reversal for SOC
    if has_soc:
        norbs_spatial = sum(norbs_per_atom)
        print("  A7. Enforcing time-reversal symmetry (SOC)...")
        sym_matrices = enforce_time_reversal(sym_matrices, norbs_spatial)

    # Replace engine matrices with symmetrized ones
    engine.real_space_matrices = sym_matrices

    # --- Part B: Re-solve with symmetrized matrices ---
    print("\n  B. Re-solving eigenvalue problems with symmetrized matrices...")
    engine.eigenvalues_list = []
    engine.eigenvectors_list = []
    engine.solve_all_kpoints(parallel=not args.no_parallel)
    print("  Eigenvalue problems re-solved")

    # Note: fix_degenerate_gauge is NOT applied here. Symmetrized eigenstates
    # in centrosymmetric systems have equal atom weight by construction (bonding/
    # antibonding). No gauge rotation can fix this. Instead, we restore the
    # original eigenvectors after band selection (see below).

    # --- Part C: Band selection ---
    # After symmetrization, the overlap structure changes and projectability
    # patterns shift. The valence bands near E_F may have lower projectability
    # than semi-core bands. Use a reduced threshold and broader energy weighting
    # to correctly capture the valence manifold.
    print("\n  C. Band selection (projectability on symmetrized data)...")

    sym_proj_threshold = getattr(args, 'proj_threshold', 0.9)
    # If user hasn't explicitly lowered the threshold, use a more permissive one
    if sym_proj_threshold >= 0.9:
        sym_proj_threshold = 0.4
        print(f"  Using reduced proj_threshold = {sym_proj_threshold} for symmetrized data")

    result = smart_select_bands(
        engine.eigenvectors_list,
        engine.S_k_list,
        engine.eigenvalues_list,
        e_fermi=engine.e_fermi,
        has_soc=has_soc,
        proj_threshold=sym_proj_threshold,
        energy_sigma=5.0,           # broader energy weighting for valence manifold
        frontier_threshold=0.29,    # capture all p-bands as frontier, exclude deep s-bands
        verbose=True,
    )

    if result.num_wann == 0:
        print("WARNING: Smart selector found no suitable bands with symmetry params.")
        print("  Falling back to standard projectability method.")
        _apply_method_projectability(engine, args, has_soc=has_soc)
        return

    user_num_wann = getattr(args, 'num_wann', None)
    if user_num_wann is not None:
        engine.num_wann = user_num_wann
        print(f"\n  User override: num_wann = {user_num_wann}")
    else:
        engine.num_wann = result.recommended_num_wann

    engine.selected_band_indices = result.selected_band_indices
    engine._num_bands_for_win = len(result.selected_band_indices)
    engine._dis_win = result.recommended_dis_win
    engine._dis_froz = result.recommended_dis_froz

    # Override frozen window: cap width to avoid freezing deep semi-core bands.
    # The symmetry method with Kramers degeneracy produces very wide frozen windows
    # from smart_select_bands because all bands get high frontier scores. Cap to
    # match the pattern of successful reference runs (e.g., bismuth_test: [-4.5, 2.0]).
    if result.recommended_dis_froz is not None:
        max_froz_half_width = 3.5  # eV from Fermi level
        froz_min = max(result.recommended_dis_froz[0], -max_froz_half_width - 1.0)
        froz_max = min(result.recommended_dis_froz[1], max_froz_half_width - 0.5)
        if froz_min != result.recommended_dis_froz[0] or froz_max != result.recommended_dis_froz[1]:
            print(f"\n  Frozen window override: [{result.recommended_dis_froz[0]:.2f}, {result.recommended_dis_froz[1]:.2f}]"
                  f" → [{froz_min:.2f}, {froz_max:.2f}] eV")
        engine._dis_froz = (froz_min, froz_max)

    # Widen outer window for maximum disentanglement freedom
    if result.recommended_dis_win is not None:
        wide_win_min = result.recommended_dis_win[0] - 10.0
        wide_win_max = result.recommended_dis_win[1] + 7.0
        engine._dis_win = (wide_win_min, wide_win_max)

    if result.recommended_dis_win is not None:
        print(f"\nDisentanglement setup:")
        print(f"  num_wann (frontier): {engine.num_wann}")
        print(f"  num_bands (total):   {engine._num_bands_for_win}")
        print(f"  dis_win  (rel E_F):  [{engine._dis_win[0]:.4f}, {engine._dis_win[1]:.4f}] eV")
        if engine._dis_froz is not None:
            print(f"  dis_froz (rel E_F):  [{engine._dis_froz[0]:.4f}, {engine._dis_froz[1]:.4f}] eV")
    else:
        print(f"\nSelected {engine.num_wann} bands, no disentanglement needed")
    print(f"Quality score: {result.quality_score:.4f}")

    # Select projection orbitals (uses symmetrized eigenvectors for weight analysis)
    proj_method = getattr(args, 'projection_method', 'weight')
    print(f"\nSelecting projection orbitals (method={proj_method})...")
    print("-" * 80)
    engine.select_projections(verbose=True, method=proj_method)

    # --- Part D: Restore original eigenvectors for AMN/MMN ---
    # Symmetrized eigenvectors have equal atom weight (centrosymmetry) → z=0 WFs.
    # Original eigenvectors retain natural site character → proper WF localization.
    # Symmetrized eigenvalues are kept for .eig file (cleaner band structure).
    print("\n  D. Restoring original eigenvectors for AMN/MMN construction...")
    engine.eigenvectors_list = original_eigenvectors
    engine.S_k_list = original_S_k
    print("  Original eigenvectors restored (site-localized)")


def _infer_projections(atoms, num_wann, has_soc):
    """Infer Wannier90 projection strings from atoms and num_wann.

    Tries to match num_wann to standard orbital sets (per atom × spinor_factor):
      - s:     1 orbital per atom
      - p:     3 orbitals per atom
      - s+p:   4 orbitals per atom (l=0;l=1, NOT sp3 hybrids)
      - d:     5 orbitals per atom
      - s+p+d: 9 orbitals per atom

    Returns None if no match found (falls back to random projections).
    """
    if atoms is None or len(atoms) == 0:
        return None

    elements = sorted(set(sym for sym, _ in atoms))
    n_atoms = len(atoms)
    sf = 2 if has_soc else 1

    # Orbitals per atom (before spinor doubling)
    orb_per_atom = num_wann // (n_atoms * sf)
    remainder = num_wann % (n_atoms * sf)

    if remainder != 0:
        return None

    # Try s-orbitals: 1 per atom
    if orb_per_atom == 1:
        projections = [f"{e}:s" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try p-orbitals: 3 per atom
    if orb_per_atom == 3:
        projections = [f"{e}:p" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try s+p: 4 per atom (use l=0;l=1 not sp3 to avoid imposing
    # tetrahedral hybridization geometry on the initial guess)
    if orb_per_atom == 4:
        projections = [f"{e}:l=0;l=1" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try d-orbitals: 5 per atom
    if orb_per_atom == 5:
        projections = [f"{e}:d" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try s+p+d: 9 per atom
    if orb_per_atom == 9:
        projections = [f"{e}:s;{e}:p;{e}:d" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    return None


def _apply_method_window(engine, args):
    """Apply window-based band selection (fallback when --window is explicit).

    If --num-wann is also specified and is less than the number of bands
    in the window, sets up disentanglement: num_bands = bands in window,
    num_wann = user override, with appropriate energy windows.
    """
    e_min, e_max = args.window

    print("\nWindow-based band selection...")
    print("-" * 80)

    result = analyze_band_window(
        engine.eigenvalues_list,
        outer_window=(e_min, e_max),
        e_fermi=engine.e_fermi,
        window_is_relative=True
    )

    num_bands_in_window = result.num_wann
    if num_bands_in_window == 0:
        print("ERROR: No bands found in the energy window!")
        print("Please adjust the energy window and try again.")
        sys.exit(1)

    engine.selected_band_indices = result.frozen_indices

    # Check for --num-wann override (disentanglement mode)
    user_num_wann = getattr(args, 'num_wann', None)
    if user_num_wann is not None and user_num_wann < num_bands_in_window:
        # Disentanglement: num_bands > num_wann
        engine.num_wann = user_num_wann
        engine._num_bands_for_win = num_bands_in_window

        # Set disentanglement windows (absolute energies)
        ef = engine.e_fermi if engine.e_fermi is not None else 0.0
        dis_win_min = ef + e_min
        dis_win_max = ef + e_max

        # Frozen window: the inner energy range containing the target bands
        # Use a narrower window around the Fermi level for the frozen states
        # Find the energy range of the num_wann bands closest to E_F
        all_evals = []
        for evals in engine.eigenvalues_list:
            selected = evals[result.frozen_indices]
            all_evals.extend(selected)
        all_evals = np.sort(all_evals)
        # The frozen window should cover the main bands we want
        # Use: [min of selected bands, E_F + small margin]
        froz_min = result.frozen_energy_range[0]
        froz_max = result.frozen_energy_range[1]
        # Tighten to roughly cover num_wann bands (heuristic: center on E_F)
        dis_froz_min = froz_min
        dis_froz_max = froz_max

        engine._dis_win = (e_min, e_max)  # Relative to E_F
        engine._dis_froz = (dis_froz_min - ef, dis_froz_max - ef)  # Relative to E_F

        print(f"Disentanglement mode:")
        print(f"  num_bands (in window):  {num_bands_in_window}")
        print(f"  num_wann  (override):   {user_num_wann}")
        print(f"  dis_win  (rel E_F):     [{e_min:.4f}, {e_max:.4f}] eV")
        print(f"  dis_froz (rel E_F):     [{dis_froz_min - ef:.4f}, {dis_froz_max - ef:.4f}] eV")
        print(f"  Energy range: [{result.frozen_energy_range[0]:.2f}, {result.frozen_energy_range[1]:.2f}] eV")
    else:
        # No disentanglement: num_bands = num_wann
        engine.num_wann = num_bands_in_window
        print(f"Selected {num_bands_in_window} bands for Wannier functions")
        print(f"  Energy range: [{result.frozen_energy_range[0]:.2f}, {result.frozen_energy_range[1]:.2f}] eV")

    # Select projection orbitals
    proj_method = getattr(args, 'projection_method', 'weight')
    print(f"\nSelecting projection orbitals (method={proj_method})...")
    print("-" * 80)
    engine.select_projections(verbose=True, method=proj_method)


def _apply_symmetry_aware_selection(engine, args, has_soc, lines):
    """Apply symmetry-aware band/projection selection for WannSym compatibility.

    Ensures the selected Wannier functions form complete orbital shells that
    close under the crystal's space group symmetry. This is required for
    Stage 3 (WannSym Reynolds operator symmetrization).

    Algorithm:
    1. Detect space group from crystal structure (via spglib)
    2. Parse orbital structure from CRYSTAL output
    3. Analyze orbital character near E_F (or use user-specified types)
    4. Compute num_wann from complete shell set
    5. Set up disentanglement with expanded band window
    6. Constrain SCDM to selected orbital subspace
    """
    from lcao_wannier.orbital_analysis import (
        analyze_orbital_type_contributions,
        compute_symmetry_aware_num_wann,
        build_orbital_mask,
        auto_select_orbital_types,
    )
    from lcao_wannier.symmetry import get_orbital_structure_from_crystal
    from lcao_wannier.parser import parse_orbital_types as parse_orb_types
    from lcao_wannier.win_file import parse_atoms_from_crystal_output

    print("\nSymmetry-aware projection selection (--symmetrize)...")
    print("-" * 80)

    # --- 1. Detect space group ---
    print("\n  Step 1: Space group detection")
    atoms_result = parse_atoms_from_crystal_output(lines)
    atoms_list, _ = atoms_result
    atom_symbols = [sym for sym, _ in atoms_list]
    atom_positions_frac = np.array([pos for _, pos in atoms_list])

    # Map symbols to atomic numbers for spglib
    _SYMBOL_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    }
    atom_numbers = np.array([_SYMBOL_TO_Z.get(s, 0) for s in atom_symbols])

    try:
        import spglib
        tolerance = getattr(args, 'sym_tolerance', 1e-5)
        cell = (engine.lattice_vectors, atom_positions_frac, atom_numbers)
        spg_info = spglib.get_spacegroup(cell, symprec=tolerance)
        symmetry = spglib.get_symmetry(cell, symprec=tolerance)
        num_ops = len(symmetry['rotations'])
        print(f"    Space group: {spg_info}")
        print(f"    Number of symmetry operations: {num_ops}")
    except Exception as e:
        print(f"    ⚠ Could not detect space group: {e}")
        print(f"    Proceeding without symmetry verification")

    # --- 2. Parse orbital structure ---
    print("\n  Step 2: Orbital structure analysis")
    atomic_info = parse_atomic_basis_info(lines)
    num_atoms = atomic_info.num_atoms
    num_basis_per_atom = atomic_info.num_basis // num_atoms

    orbital_types_dict = parse_orb_types(lines, has_soc=has_soc, num_atoms=num_atoms)
    orbital_structure = get_orbital_structure_from_crystal(
        orbital_types_dict, num_basis_per_atom, num_atoms
    )

    for i, shells in enumerate(orbital_structure):
        print(f"    Atom {i} ({atom_symbols[i]}): shells = {shells}")

    # --- 3. Select orbital types ---
    print("\n  Step 3: Orbital type selection")
    symm_orbitals = getattr(args, 'symm_orbitals', 'auto')

    if symm_orbitals == 'auto':
        # Auto-detect from band structure near E_F
        type_contributions = analyze_orbital_type_contributions(
            engine.eigenvectors_list,
            engine.S_k_list,
            engine.eigenvalues_list,
            orbital_types_dict,
            e_fermi=engine.e_fermi,
            has_soc=has_soc,
            verbose=True,
        )
        selected_types = auto_select_orbital_types(type_contributions, verbose=True)
    else:
        # Parse user-specified types: 'p' → ['p'], 'sp' → ['s', 'p'], 'spd' → ['s', 'p', 'd']
        type_order = ['s', 'p', 'd', 'f', 'g']
        selected_types = [t for t in type_order if t in symm_orbitals.lower()]
        if not selected_types:
            print(f"    ERROR: Could not parse orbital types from '{symm_orbitals}'")
            sys.exit(1)
        print(f"    User-specified orbital types: {selected_types}")

    # --- 4. Compute num_wann ---
    print("\n  Step 4: Computing num_wann for complete shells")
    num_wann, wannier_orbital_structure = compute_symmetry_aware_num_wann(
        selected_types, orbital_structure, has_soc=has_soc, verbose=True
    )
    engine.num_wann = num_wann

    # Store orbital structure for Stage 3 compatibility
    engine._wannier_orbital_structure = wannier_orbital_structure

    # --- 5. Set up band selection with disentanglement ---
    print("\n  Step 5: Band selection with disentanglement")

    # Use smart_select_bands to get recommended windows
    result = smart_select_bands(
        engine.eigenvectors_list,
        engine.S_k_list,
        engine.eigenvalues_list,
        e_fermi=engine.e_fermi,
        has_soc=has_soc,
        proj_threshold=args.proj_threshold,
        verbose=True,
    )

    engine._dis_win = result.recommended_dis_win
    engine._dis_froz = result.recommended_dis_froz
    ef = engine.e_fermi if engine.e_fermi is not None else 0.0

    # Determine band selection — must have num_bands >= num_wann + 2
    # for proper disentanglement
    target_num_bands = max(num_wann + 2, len(result.selected_band_indices))

    # Start from the recommended dis_win if available
    if result.recommended_dis_win is not None:
        win_min = ef + result.recommended_dis_win[0]
        win_max = ef + result.recommended_dis_win[1]
    else:
        # Fallback: window around E_F
        win_min = ef - 15.0
        win_max = ef + 5.0

    # Count bands in window at each k-point
    min_bands = None
    for evals in engine.eigenvalues_list:
        count = np.sum((evals >= win_min) & (evals <= win_max))
        if min_bands is None or count < min_bands:
            min_bands = count

    # If the window doesn't capture enough bands, expand it
    if min_bands < target_num_bands:
        print(f"    dis_win has {min_bands} bands, need {target_num_bands} — expanding window...")
        # Expand symmetrically in 1 eV steps until we have enough
        expansion = 0.0
        while min_bands < target_num_bands and expansion < 30.0:
            expansion += 1.0
            w_min = win_min - expansion
            w_max = win_max + expansion
            min_bands = None
            for evals in engine.eigenvalues_list:
                count = np.sum((evals >= w_min) & (evals <= w_max))
                if min_bands is None or count < min_bands:
                    min_bands = count
        win_min = w_min
        win_max = w_max
        # Update dis_win to reflect expanded window
        engine._dis_win = (win_min - ef, win_max - ef)
        print(f"    Expanded dis_win: [{win_min - ef:.2f}, {win_max - ef:.2f}] eV rel E_F")

    # Select band indices from the window
    evals_gamma = engine.eigenvalues_list[0]
    in_window = np.where(
        (evals_gamma >= win_min) & (evals_gamma <= win_max)
    )[0]
    num_bands_final = min(min_bands, len(in_window))
    engine.selected_band_indices = in_window[:num_bands_final]
    engine._num_bands_for_win = num_bands_final

    print(f"    Band selection: {num_bands_final} bands")
    print(f"    Band indices: {engine.selected_band_indices}")

    if engine._num_bands_for_win > engine.num_wann:
        print(f"\n  Disentanglement setup:")
        print(f"    num_wann (target):   {engine.num_wann}")
        print(f"    num_bands (total):   {engine._num_bands_for_win}")
        if engine._dis_win is not None:
            print(f"    dis_win  (rel E_F):  [{engine._dis_win[0]:.4f}, {engine._dis_win[1]:.4f}] eV")
        if engine._dis_froz is not None:
            print(f"    dis_froz (rel E_F):  [{engine._dis_froz[0]:.4f}, {engine._dis_froz[1]:.4f}] eV")
    else:
        print(f"\n  No disentanglement: num_wann = num_bands = {engine.num_wann}")

    # --- 6. Constrain SCDM to selected orbital subspace ---
    print("\n  Step 6: SCDM projection selection (constrained)")

    orbital_mask = build_orbital_mask(
        orbital_types_dict, selected_types, engine.num_orbitals,
        has_soc=has_soc, verbose=True,
    )

    # Force SCDM method for symmetry-constrained selection
    engine.select_projections(verbose=True, method='scdm', orbital_mask=orbital_mask)

    # Override num_iter for convergence
    if engine._override_num_iter is None:
        engine._override_num_iter = 5000

    print(f"\n  ✓ Symmetry-aware selection complete:")
    print(f"    Orbital types: {selected_types}")
    print(f"    num_wann: {num_wann}")
    print(f"    num_bands: {engine._num_bands_for_win}")


def stage1_create_win(args):
    """
    Stage 1: Parse LCAO output and create .win file only.

    This prepares the Wannier90 input file so the user can run
    'wannier90.x -pp seedname' to generate the .nnkp file.
    """
    print("=" * 80)
    print("STAGE 1: Creating Wannier90 Parameter File (.win)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Parse CRYSTAL output
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    # Apply user --k-grid override (for memory-limited systems or 2D slabs)
    if getattr(args, 'k_grid', None) is not None:
        original_kgrid = params.k_grid
        params.k_grid = tuple(args.k_grid)
        print(f"  Overriding k-grid: {original_kgrid} -> {params.k_grid} "
              f"(user --k-grid)")
    # Apply user --fermi-energy override (bypass SPINLOCK/level-shift-corrupted value)
    if getattr(args, 'fermi_energy', None) is not None:
        original_fermi = params.fermi_energy
        params.fermi_energy = float(args.fermi_energy)
        print(f"  Overriding Fermi energy: {original_fermi} eV -> "
              f"{params.fermi_energy} eV (user --fermi-energy)")
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed calculation parameters:")
    if params.fermi_energy is not None:
        print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    else:
        print(f"  Fermi energy: Not found (will estimate later)")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Number of AOs: {params.num_ao}")

    # Use SOC detection from parser (TWO-COMPONENT SCF marker)
    has_soc = params.has_soc

    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # If SOC is detected, we need to double the number of orbitals for spinors
    if has_soc:
        print(f"  → Doubling orbitals for spinors: {params.num_ao} → {params.num_ao * 2}")

    # Organize matrices by R-vector
    print("\nStep 2: Organizing matrices...")
    print("-" * 80)

    H_R_dict = {}
    S_R_dict = {}
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        mat_type = mat_info['type']
        data = mat_info['data']

        if mat_type == 'overlap':
            S_R_dict[R] = data
        elif mat_type == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data

    num_basis = params.num_ao if params.num_ao else list(S_R_dict.values())[0].shape[0]
    print(f"✓ Organized matrices")
    print(f"  Basis size: {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Create spin-block matrices if SOC
    print("\nStep 3: Creating matrices...")
    print("-" * 80)

    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        matrix_size = H_full_list[0][1].shape[0]
        print(f"✓ Created {matrix_size}×{matrix_size} SOC matrices")
    else:
        # For non-SOC, symmetrize lower-triangular raw matrices using R/-R pairs
        H_full_list, S_full_list = create_nonsoc_full_matrices(
            H_R_dict, S_R_dict, lattice_vectors_list
        )
        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} symmetrized matrices (no SOC)")

    # Prepare real-space matrices in the format engine expects
    print("\nStep 4: Preparing real-space matrices...")
    print("-" * 80)

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    print(f"✓ Prepared {len(real_space_matrices)} R-vectors")

    # Initialize engine (window only needed for window-based fallback)
    print("\nStep 5: Initializing Wannier90 engine...")
    print("-" * 80)

    # Determine energy window (used as fallback or with explicit --window)
    if args.window:
        e_min, e_max = args.window
    else:
        e_min, e_max = -5.0, 3.0  # Default window

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=args.seedname,
        num_wann=None,
        outer_window=(e_min, e_max),
        e_fermi=params.fermi_energy,
        window_is_relative=True
    )

    print("✓ Engine initialized")

    # Parse atomic basis information for phase-corrected MMN
    print("\nParsing atomic basis information...")
    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
        atomic_info = parse_atomic_basis_info(lines)
        engine.atom_positions = atomic_info.atom_positions

        # For SOC systems, double the basis_atom_map (spin up + spin down)
        num_orbitals = engine.num_orbitals
        if num_orbitals == 2 * atomic_info.num_basis:
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,
                atomic_info.basis_atom_map
            ])
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions (SOC: doubled basis map)")
        else:
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction")

    # Solve eigenvalue problems
    print("\nStep 6: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel)
    print("✓ Eigenvalue problems solved")

    # Sanity-check parsed Fermi vs HOMO from electron count. Auto-falls
    # back to the electron-count estimate with a loud warning if the two
    # disagree by > FERMI_FRAME_TOLERANCE_EV (CRYSTAL SPINLOCK/shift
    # frame-mismatch detection).
    _sanity_check_fermi_energy(engine, params, args, stage_name="Stage 1")

    # Estimate Fermi energy if needed
    if params.fermi_energy is None:
        print("\nEstimating Fermi energy...")
        if params.num_electrons is not None:
            fermi_energy = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=params.num_electrons,
                method='auto'
            )
            engine.e_fermi = fermi_energy
            print(f"✓ Estimated Fermi energy: {fermi_energy:.6f} eV")
        else:
            print("⚠ Cannot estimate Fermi energy (num_electrons not found)")
            print("  Using E_F = 0.0 eV")
            engine.e_fermi = 0.0

    # Method dispatch for band selection
    print(f"\nStep 7: Band selection (method={args.method})...")
    print("-" * 80)

    if getattr(args, 'symmetrize', False):
        _apply_symmetry_aware_selection(engine, args, has_soc, lines)
    elif args.method == 'symmetry':
        _apply_method_symmetry(engine, args, has_soc, params, lines)
    elif args.method == 'direct':
        _apply_method_direct(engine, args, has_soc, params)
    elif args.method in ('pdwf', 'auto'):
        _apply_method_pdwf(engine, args, has_soc, lines)
    elif args.method == 'projectability':
        if args.window:
            print(f"Using explicit energy window: [{e_min:.2f}, {e_max:.2f}] eV")
            _apply_method_window(engine, args)
        else:
            _apply_method_projectability(engine, args, has_soc=has_soc)

    # Parse atoms from CRYSTAL output if available
    print("\nStep 8: Extracting atomic positions...")
    print("-" * 80)

    try:
        with open(args.input, 'r') as f:
            crystal_lines = f.readlines()
        atoms_result = parse_atoms_from_crystal_output(crystal_lines)
        atoms = atoms_result[0]  # Returns (atoms_list, lattice_vectors)
        print(f"✓ Found {len(atoms)} atoms")
        for symbol, pos in atoms:
            print(f"  {symbol}: {pos}")
    except Exception as e:
        print(f"⚠ Could not parse atoms: {e}")
        print("  Will create .win without atoms block")
        atoms = None

    # Write .win file only
    print(f"\nStep 9: Writing {args.seedname}.win file...")
    print("-" * 80)

    # Prepare projections — try to auto-infer if not explicitly provided
    projections = args.projections if args.projections else None
    if projections is None and atoms is not None:
        projections = _infer_projections(atoms, engine.num_wann, has_soc)

    # Set num_iter for projectability method (5000 iterations for proper convergence)
    if args.method == 'projectability' and engine._override_num_iter is None:
        engine._override_num_iter = 5000

    # Set parameters for symmetry method
    if args.method == 'symmetry':
        if engine._override_num_iter is None:
            engine._override_num_iter = 5000

    # Auto-detect kpoint path for band structure plots
    kpoint_path = None
    if args.bands_plot:
        from lcao_wannier.win_file import (
            KPATH_HEXAGONAL_2D, KPATH_SIMPLE_CUBIC, KPATH_FCC, KPATH_BCC
        )
        lv = engine.lattice_vectors
        a1_len = np.linalg.norm(lv[0])
        a2_len = np.linalg.norm(lv[1])
        a3_len = np.linalg.norm(lv[2])

        if a3_len > 10 * a1_len:
            # 2D system: large vacuum along a3
            kpoint_path = KPATH_HEXAGONAL_2D
            print(f"  Auto-detected 2D hexagonal lattice -> Gamma-M-K band path")
        else:
            # 3D system: detect lattice type from angles and lengths
            angles = []
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                cos_a = np.dot(lv[i], lv[j]) / (np.linalg.norm(lv[i]) * np.linalg.norm(lv[j]))
                angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
            alpha, beta, gamma = angles
            lengths = [a1_len, a2_len, a3_len]
            equal_lengths = (abs(lengths[0] - lengths[1]) < 0.01 * lengths[0] and
                             abs(lengths[1] - lengths[2]) < 0.01 * lengths[1])
            all_90 = all(abs(a - 90.0) < 1.0 for a in [alpha, beta, gamma])

            if equal_lengths and all_90:
                kpoint_path = KPATH_SIMPLE_CUBIC
                print(f"  Auto-detected simple cubic lattice -> Gamma-X-M-R band path")
            elif equal_lengths and not all_90:
                # Could be FCC or BCC (rhombohedral primitive cell)
                avg_angle = np.mean([alpha, beta, gamma])
                if avg_angle < 80:
                    kpoint_path = KPATH_FCC
                    print(f"  Auto-detected FCC-like lattice -> Gamma-X-W-K-L band path")
                else:
                    kpoint_path = KPATH_BCC
                    print(f"  Auto-detected BCC-like lattice -> Gamma-H-N-P band path")

            if kpoint_path is None:
                print(f"  Could not auto-detect lattice type for band path")
                print(f"  Lattice lengths: {lengths}, angles: {angles}")

    # Write only the .win file
    engine.write_files(
        verbose=True,
        write_win=True,
        use_nnkp=False,  # Don't use .nnkp in stage 1
        atoms=atoms,
        projections=projections,
        spinors=has_soc,
        bands_plot=args.bands_plot,
        kpoint_path=kpoint_path,
    )

    # Delete the .eig, .amn, .mmn files if they were created
    # (write_files creates them by default, but we only want .win in stage 1)
    for ext in ['.eig', '.amn', '.mmn']:
        filepath = f"{args.seedname}{ext}"
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  (Removed premature {filepath})")

    print()
    print("=" * 80)
    print("STAGE 1 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.win")
    print()
    print("NEXT STEP:")
    print(f"  Run Wannier90 preprocessing to generate neighbor information:")
    print(f"  → wannier90.x -pp {args.seedname}")
    print()
    print(f"  This will create {args.seedname}.nnkp")
    print()
    print("Then proceed to Stage 2:")
    print(f"  python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
    print("=" * 80)


def stage2_create_data_files(args):
    """
    Stage 2: Read .nnkp and create .eig, .amn, .mmn files.

    This uses the neighbor information from Wannier90's preprocessing
    to generate data files with the exact neighbor structure expected.
    """
    print("=" * 80)
    print("STAGE 2: Creating Wannier90 Data Files (.eig, .amn, .mmn)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check that .nnkp file exists
    nnkp_file = f"{args.seedname}.nnkp"
    if not os.path.exists(nnkp_file):
        print(f"ERROR: {nnkp_file} not found!")
        print()
        print("You must run Stage 1 first, then run Wannier90 preprocessing:")
        print(f"  1. python {sys.argv[0]} --stage 1 --input {args.input} --seedname {args.seedname}")
        print(f"  2. wannier90.x -pp {args.seedname}")
        print(f"  3. python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
        print()
        sys.exit(1)

    # Check that .win file exists
    win_file = f"{args.seedname}.win"
    if not os.path.exists(win_file):
        print(f"ERROR: {win_file} not found!")
        print("Please run Stage 1 first to create the .win file.")
        sys.exit(1)

    print(f"✓ Found {nnkp_file}")
    print(f"✓ Found {win_file}")
    print()

    # Parse CRYSTAL output (same as stage 1)
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    # Apply user --k-grid override (for memory-limited systems or 2D slabs)
    if getattr(args, 'k_grid', None) is not None:
        original_kgrid = params.k_grid
        params.k_grid = tuple(args.k_grid)
        print(f"  Overriding k-grid: {original_kgrid} -> {params.k_grid} "
              f"(user --k-grid)")
    # Apply user --fermi-energy override (bypass SPINLOCK/level-shift-corrupted value)
    if getattr(args, 'fermi_energy', None) is not None:
        original_fermi = params.fermi_energy
        params.fermi_energy = float(args.fermi_energy)
        print(f"  Overriding Fermi energy: {original_fermi} eV -> "
              f"{params.fermi_energy} eV (user --fermi-energy)")
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed calculation parameters:")
    if params.fermi_energy is not None:
        print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    else:
        print(f"  Fermi energy: Not found (will estimate later)")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Number of AOs: {params.num_ao}")

    # Use SOC detection from parser (TWO-COMPONENT SCF marker)
    has_soc = params.has_soc

    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # If SOC is detected, we need to double the number of orbitals for spinors
    if has_soc:
        print(f"  → Doubling orbitals for spinors: {params.num_ao} → {params.num_ao * 2}")

    # Organize matrices by R-vector
    print("\nStep 2: Organizing matrices...")
    print("-" * 80)

    H_R_dict = {}
    S_R_dict = {}
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        mat_type = mat_info['type']
        data = mat_info['data']

        if mat_type == 'overlap':
            S_R_dict[R] = data
        elif mat_type == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data

    num_basis = params.num_ao if params.num_ao else list(S_R_dict.values())[0].shape[0]
    print(f"✓ Organized matrices")
    print(f"  Basis size: {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Create spin-block matrices if SOC
    print("\nStep 3: Creating matrices...")
    print("-" * 80)

    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        matrix_size = H_full_list[0][1].shape[0]
        print(f"✓ Created {matrix_size}×{matrix_size} SOC matrices")
    else:
        # For non-SOC, symmetrize lower-triangular raw matrices using R/-R pairs
        H_full_list, S_full_list = create_nonsoc_full_matrices(
            H_R_dict, S_R_dict, lattice_vectors_list
        )
        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} symmetrized matrices (no SOC)")

    # Prepare real-space matrices
    print("\nStep 4: Preparing real-space matrices...")
    print("-" * 80)

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    print(f"✓ Prepared {len(real_space_matrices)} R-vectors")

    # Read num_wann and num_bands from .win file if it exists
    print("\nStep 5: Reading parameters from .win file...")
    print("-" * 80)

    num_wann_from_win = None
    num_bands_from_win = None
    dis_froz_min_from_win = None
    dis_froz_max_from_win = None
    win_file = f"{args.seedname}.win"

    if os.path.exists(win_file):
        with open(win_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('num_wann'):
                    try:
                        num_wann_from_win = int(line.split('=')[1].strip())
                        print(f"  Found num_wann = {num_wann_from_win}")
                    except:
                        pass
                elif line.startswith('num_bands'):
                    try:
                        num_bands_from_win = int(line.split('=')[1].strip())
                        print(f"  Found num_bands = {num_bands_from_win}")
                    except:
                        pass
                elif line.startswith('dis_froz_min'):
                    try:
                        dis_froz_min_from_win = float(line.split('=')[1].strip())
                        print(f"  Found dis_froz_min = {dis_froz_min_from_win}")
                    except:
                        pass
                elif line.startswith('dis_froz_max'):
                    try:
                        dis_froz_max_from_win = float(line.split('=')[1].strip())
                        print(f"  Found dis_froz_max = {dis_froz_max_from_win}")
                    except:
                        pass
        print(f"✓ Read parameters from {win_file}")
    else:
        print(f"⚠ Warning: {win_file} not found, using automatic band selection")

    # Determine energy window (same as stage 1)
    print("\nStep 6: Determining energy window...")
    print("-" * 80)

    if args.window:
        e_min, e_max = args.window
        print(f"Using user-specified window: [{e_min:.2f}, {e_max:.2f}] eV")
    else:
        e_min, e_max = -5.0, 3.0
        print(f"Using default window: [{e_min:.2f}, {e_max:.2f}] eV (relative to E_F)")

    # Initialize engine
    print("\nStep 7: Initializing Wannier90 engine...")
    print("-" * 80)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=args.seedname,
        num_wann=num_wann_from_win,  # Use value from .win file
        outer_window=(e_min, e_max),
        e_fermi=params.fermi_energy,
        window_is_relative=True
    )

    print("✓ Engine initialized")

    # Parse atomic basis information for phase-corrected MMN
    print("\nParsing atomic basis information...")
    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
        atomic_info = parse_atomic_basis_info(lines)
        engine.atom_positions = atomic_info.atom_positions

        # For SOC systems, double the basis_atom_map (spin up + spin down)
        num_orbitals = engine.num_orbitals
        if num_orbitals == 2 * atomic_info.num_basis:
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,
                atomic_info.basis_atom_map
            ])
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions (SOC: doubled basis map)")
        else:
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction (may cause negative spreads!)")

    # Set up site_symmetry attributes if requested
    if getattr(args, 'site_symmetry', False):
        try:
            from lcao_wannier.parser import parse_orbital_types
            from lcao_wannier.symmetry import get_orbital_structure_from_crystal

            # Get fractional positions and atomic numbers
            A_inv = np.linalg.inv(lattice_vectors.T)
            atom_positions_frac = (A_inv @ atomic_info.atom_positions.T).T
            # Wrap to [0, 1)
            atom_positions_frac -= np.floor(atom_positions_frac)
            engine.atom_positions_frac = atom_positions_frac

            # Get atomic numbers from symbols
            _SYMBOL_TO_Z = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
                'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
                'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
                'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
                'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
                'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
                'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
                'Ba': 56, 'La': 57, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
                'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
                'Pb': 82, 'Bi': 83,
            }
            engine.atom_numbers = np.array([
                _SYMBOL_TO_Z.get(s, 0) for s in atomic_info.atom_symbols
            ])

            # Get orbital types per atom
            orbital_types_dict = parse_orbital_types(lines)
            num_basis_per_atom = []
            for iatom in range(atomic_info.num_atoms):
                count = int(np.sum(atomic_info.basis_atom_map == iatom))
                num_basis_per_atom.append(count)

            orb_per_atom = get_orbital_structure_from_crystal(
                orbital_types_dict, num_basis_per_atom, atomic_info.num_atoms
            )

            # Validate: total orbitals from structure must match actual basis
            from lcao_wannier.symmetry import _orbital_dim
            total_from_structure = sum(
                _orbital_dim(ot) for atom_orbs in orb_per_atom for ot in atom_orbs
            )
            if total_from_structure != atomic_info.num_basis:
                # Orbital dict may be misaligned for equivalent atoms
                # Copy structure from first atom of same element
                print(f"  ⚠ Orbital structure mismatch ({total_from_structure} vs "
                      f"{atomic_info.num_basis}), fixing via equivalent-atom copying...")
                fixed = [None] * atomic_info.num_atoms
                for iatom in range(atomic_info.num_atoms):
                    sym = atomic_info.atom_symbols[iatom]
                    nbas = num_basis_per_atom[iatom]
                    # Find first atom of same element with same basis count
                    ref = None
                    for jatom in range(iatom):
                        if (atomic_info.atom_symbols[jatom] == sym
                                and num_basis_per_atom[jatom] == nbas
                                and fixed[jatom] is not None):
                            ref = jatom
                            break
                    if ref is not None:
                        fixed[iatom] = list(fixed[ref])
                    else:
                        fixed[iatom] = orb_per_atom[iatom]
                orb_per_atom = fixed
                # Re-validate
                total_fixed = sum(
                    _orbital_dim(ot) for atom_orbs in orb_per_atom for ot in atom_orbs
                )
                if total_fixed != atomic_info.num_basis:
                    raise RuntimeError(
                        f"Cannot fix orbital structure: {total_fixed} != {atomic_info.num_basis}"
                    )
                print(f"  ✓ Fixed orbital structure: {total_fixed} orbitals")

            engine.orbital_types_per_atom = orb_per_atom
            engine.has_soc = has_soc
            print(f"✓ Set up site_symmetry data: {atomic_info.num_atoms} atoms, "
                  f"orbital types: {engine.orbital_types_per_atom}")
        except Exception as e:
            print(f"ERROR: Cannot set up site_symmetry: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Solve eigenvalue problems
    print("\nStep 7: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel)
    print("✓ Eigenvalue problems solved")

    # Sanity-check parsed Fermi vs HOMO from electron count. Auto-falls
    # back to the electron-count estimate with a loud warning if the two
    # disagree by > FERMI_FRAME_TOLERANCE_EV (CRYSTAL SPINLOCK/shift
    # frame-mismatch detection).
    _sanity_check_fermi_energy(engine, params, args, stage_name="Stage 2")

    # Estimate Fermi energy if needed
    if params.fermi_energy is None:
        print("\nEstimating Fermi energy...")
        if params.num_electrons is not None:
            fermi_energy = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=params.num_electrons,
                method='auto'
            )
            engine.e_fermi = fermi_energy
            print(f"✓ Estimated Fermi energy: {fermi_energy:.6f} eV")
        else:
            print("⚠ Cannot estimate Fermi energy (num_electrons not found)")
            print("  Using E_F = 0.0 eV")
            engine.e_fermi = 0.0

    # Method dispatch for band selection
    print(f"\nStep 8: Band selection (method={args.method})...")
    print("-" * 80)

    if getattr(args, 'symmetrize', False):
        _apply_symmetry_aware_selection(engine, args, has_soc, lines)
    elif args.method == 'symmetry':
        _apply_method_symmetry(engine, args, has_soc, params, lines)
    elif args.method in ('pdwf', 'auto'):
        _apply_method_pdwf(engine, args, has_soc, lines)
    elif args.method == 'direct':
        _apply_method_direct(engine, args, has_soc, params)
    elif args.method == 'projectability':
        if args.window:
            print(f"Using explicit energy window: [{e_min:.2f}, {e_max:.2f}] eV")
            _apply_method_window(engine, args)
        elif num_wann_from_win is not None:
            # .win file exists with num_wann — use projectability but
            # verify consistency with .win parameters
            _apply_method_projectability(engine, args, has_soc=has_soc)
            if engine.num_wann != num_wann_from_win:
                print(f"\n⚠ Note: Projectability selected {engine.num_wann} bands, "
                      f"but .win file has num_wann = {num_wann_from_win}")
                print(f"  Using projectability result ({engine.num_wann} bands)")
        else:
            _apply_method_projectability(engine, args, has_soc=has_soc)

    # Write data files using .nnkp neighbors
    print(f"\nStep 9: Writing data files using {nnkp_file} neighbors...")
    print("-" * 80)

    # Prepare AMN symmetrization data if requested
    amn_sym_kwargs = {}
    if getattr(args, 'symmetrize_amn', False):
        try:
            atoms_result = parse_atoms_from_crystal_output(lines)
            atoms_list_sym, _ = atoms_result
            atom_syms = [sym for sym, _ in atoms_list_sym]
            _SYMBOL_TO_Z = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
                'Ce': 58, 'Bi': 83,
            }
            amn_sym_kwargs = {
                'symmetrize_amn': True,
                'atom_positions_frac': np.array([pos for _, pos in atoms_list_sym]),
                'atom_numbers': np.array([_SYMBOL_TO_Z.get(s, 0) for s in atom_syms]),
            }
            print(f"  AMN symmetrization enabled ({len(atoms_list_sym)} atoms)")
        except Exception as e:
            print(f"  ⚠ Cannot symmetrize AMN: {e}")

    # Write only the data files, not .win (already exists)
    engine.write_files(
        verbose=True,
        write_win=False,  # Don't overwrite .win
        use_nnkp=True,    # Use .nnkp neighbors (CRITICAL!)

        **amn_sym_kwargs,

        site_symmetry=getattr(args, 'site_symmetry', False),

    )

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.eig")
    print(f"✓ Created: {args.seedname}.amn")
    print(f"✓ Created: {args.seedname}.mmn")
    if getattr(args, 'site_symmetry', False):
        print(f"✓ Created: {args.seedname}.dmn (site symmetry)")
    print()
    print("All files generated with correct neighbor structure from .nnkp!")
    if getattr(args, 'site_symmetry', False):
        print("Note: site_symmetry = .true. should be added to the .win file")
        print("  (or re-run Stage 1 with --site-symmetry)")
    print()
    print("NEXT STEP:")
    print(f"  Run Wannier90 to generate maximally localized Wannier functions:")
    print(f"  → wannier90.x {args.seedname}")
    print()
    print("=" * 80)


def stage3_symmetrize_hr(args):
    """
    Stage 3: Symmetrize wannier90_hr.dat using crystal symmetry.

    Uses the Reynolds operator to enforce all space group symmetries on the
    real-space Hamiltonian produced by Wannier90. This is a post-processing
    step that improves the symmetry properties of the tight-binding model.

    Requires:
      - CRYSTAL output file (for crystal structure and orbital types)
      - wannier90_hr.dat (from Wannier90 run after Stage 2)
    """
    from lcao_wannier.wannsym import symmetrize_hr, HamiltonianData, WannSymConfig
    from lcao_wannier.symmetry import get_orbital_structure_from_crystal

    print("=" * 80)
    print("STAGE 3: Symmetrize Wannier90 Hamiltonian (wannier90_hr.dat)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input files
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    hr_file = args.hr_file or f"{args.seedname}_hr.dat"
    if not os.path.exists(hr_file):
        print(f"ERROR: HR file not found: {hr_file}")
        print()
        print("You must run Wannier90 first (after Stage 2) to produce the HR file.")
        print(f"  Expected: {hr_file}")
        print()
        print("If the file has a different name, use --hr-file to specify it.")
        sys.exit(1)

    print(f"  HR file: {hr_file}")
    print()

    # --- Step 1: Parse CRYSTAL output for crystal structure ---
    print("Step 1: Parsing crystal structure from CRYSTAL output...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    _, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    has_soc = params.has_soc
    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # Extract atom positions and types
    # Use parse_atoms_from_crystal_output for fractional coordinates
    # (parse_atomic_basis_info returns Cartesian, which breaks spglib)
    atoms_result = parse_atoms_from_crystal_output(lines)
    atoms_list, _ = atoms_result
    atom_symbols = [sym for sym, _ in atoms_list]
    atom_positions_frac = np.array([pos for _, pos in atoms_list])
    num_atoms = len(atom_symbols)

    # Also get basis info for orbital structure
    atomic_info = parse_atomic_basis_info(lines)

    # Map symbols to atomic numbers
    _SYMBOL_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    }
    atom_numbers = np.array([_SYMBOL_TO_Z.get(s, 0) for s in atom_symbols])

    print(f"  Structure: {num_atoms} atoms ({', '.join(sorted(set(atom_symbols)))})")
    print(f"  Lattice vectors shape: {lattice_vectors.shape}")

    # --- Step 2: Parse orbital structure ---
    print("\nStep 2: Parsing orbital structure...")
    print("-" * 80)

    orbital_types_dict = parse_orbital_types(lines, has_soc=False, num_atoms=num_atoms)

    if not orbital_types_dict:
        print("ERROR: Could not parse orbital types from CRYSTAL output.")
        print("  The orbital structure is needed to build representation matrices.")
        sys.exit(1)

    # Build per-atom orbital shell list
    num_basis_per_atom = max(orbital_types_dict.keys()) // num_atoms
    orbital_structure = get_orbital_structure_from_crystal(
        orbital_types_dict, num_basis_per_atom, num_atoms
    )

    print(f"  Orbital structure per atom:")
    for i, shells in enumerate(orbital_structure):
        print(f"    Atom {i} ({atom_symbols[i]}): {shells}")

    # --- Step 3: Determine which atoms are in the Wannier model ---
    # The CRYSTAL output has the full crystal structure, but the Wannier HR
    # may only contain a subset of atoms. We need to figure out which atoms
    # are included by checking the HR matrix dimension.
    print("\nStep 3: Loading Wannier90 Hamiltonian...")
    print("-" * 80)

    hr_data = HamiltonianData.from_file(hr_file)
    print(f"  Loaded: {hr_data.norbs} orbitals, {hr_data.nrpt} R-points")

    # Figure out which atoms are in the Wannier model
    # Compute the number of orbitals per atom (including SOC doubling)
    from lcao_wannier.symmetry import _orbital_dim
    norbs_per_atom = []
    for atom_orbs in orbital_structure:
        n = sum(_orbital_dim(t) for t in atom_orbs)
        norbs_per_atom.append(n)

    spinor_factor = 2 if has_soc else 1
    total_orbs_all_atoms = sum(norbs_per_atom) * spinor_factor

    if hr_data.norbs == total_orbs_all_atoms:
        # All atoms included — use full orbital structure
        wannier_orbital_structure = orbital_structure
        print(f"  All {num_atoms} atoms included in Wannier model "
              f"({total_orbs_all_atoms} orbitals)")
    else:
        # Subset of atoms — try to identify which ones
        # Check if num_wann matches a subset
        print(f"  HR has {hr_data.norbs} orbitals, crystal has {total_orbs_all_atoms}")
        print(f"  Attempting to identify Wannier atom subset...")

        # Try to read num_wann from .win file for hints
        win_file = f"{args.seedname}.win"
        wannier_orbital_structure = None

        if os.path.exists(win_file):
            # Look for projection block to identify atoms
            print(f"  Reading {win_file} for projection info...")

        # Try to parse projections from .win file to determine orbital subset
        if wannier_orbital_structure is None and os.path.exists(win_file):
            with open(win_file, 'r') as f:
                win_lines = f.readlines()

            # Parse "begin projections" ... "end projections" block
            in_proj = False
            proj_lines = []
            for line in win_lines:
                stripped = line.strip()
                if stripped.lower() == 'begin projections':
                    in_proj = True
                    continue
                elif stripped.lower() == 'end projections':
                    in_proj = False
                    continue
                if in_proj and stripped and not stripped.startswith('!'):
                    proj_lines.append(stripped)

            if proj_lines:
                # Parse projections like "BI:p", "BI:sp3", "BI:s;p;d"
                # Map to orbital types per atom
                # All atoms of the same element get the same projection
                proj_per_element = {}
                for pline in proj_lines:
                    if ':' in pline:
                        elem, orb_spec = pline.split(':', 1)
                        elem = elem.strip().upper()
                        # Parse orbital spec: "p", "sp3", "s;p;d", "l=0;l=1"
                        orbitals = []
                        orb_spec = orb_spec.strip()
                        if ';' in orb_spec:
                            parts = orb_spec.split(';')
                        else:
                            parts = [orb_spec]
                        for part in parts:
                            part = part.strip()
                            if part in ('s', 'p', 'd', 'f'):
                                orbitals.append(part)
                            elif 'l=0' in part:
                                orbitals.append('s')
                            elif 'l=1' in part:
                                orbitals.append('p')
                            elif 'l=2' in part:
                                orbitals.append('d')
                            elif 'l=3' in part:
                                orbitals.append('f')
                            elif part in ('sp', 'sp2', 'sp3', 'sp3d', 'sp3d2'):
                                # Hybrid orbitals — decompose
                                if 's' in part:
                                    orbitals.append('s')
                                if 'p' in part:
                                    orbitals.append('p')
                                if 'd' in part:
                                    orbitals.append('d')
                        if orbitals:
                            proj_per_element[elem] = orbitals

                if proj_per_element:
                    print(f"  Parsed projections from .win: {proj_per_element}")
                    # Build wannier_orbital_structure for each atom
                    wannier_orbital_structure = []
                    for sym in atom_symbols:
                        elem_key = sym.upper()
                        if elem_key in proj_per_element:
                            wannier_orbital_structure.append(proj_per_element[elem_key])
                        else:
                            # Atom not in projections — skip
                            continue

                    # Verify dimension matches
                    from lcao_wannier.symmetry import _orbital_dim
                    total_check = sum(
                        sum(_orbital_dim(t) for t in shells)
                        for shells in wannier_orbital_structure
                    ) * spinor_factor
                    if total_check == hr_data.norbs:
                        print(f"  ✓ Matched: {total_check} orbitals from projections")
                    else:
                        print(f"  Projection-based count ({total_check}) != HR ({hr_data.norbs})")
                        wannier_orbital_structure = None

        # Fallback: assume all atoms if orbital counts divide evenly
        # This works when all atoms have the same orbital structure
        if wannier_orbital_structure is None:
            # Check if hr_data.norbs matches some subset
            cumulative = 0
            subset_atoms = []
            for i, n in enumerate(norbs_per_atom):
                cumulative += n * spinor_factor
                subset_atoms.append(i)
                if cumulative == hr_data.norbs:
                    break

            if cumulative == hr_data.norbs:
                wannier_orbital_structure = [orbital_structure[i] for i in subset_atoms]
                print(f"  Identified {len(subset_atoms)} Wannier atoms: "
                      f"{[atom_symbols[i] for i in subset_atoms]}")
            else:
                print(f"  WARNING: Cannot match HR dimension ({hr_data.norbs}) "
                      f"to crystal structure atoms.")
                print(f"  Using full orbital structure — symmetrization may fail.")
                wannier_orbital_structure = orbital_structure

    # --- Step 4: Symmetrize ---
    print("\nStep 4: Symmetrizing Hamiltonian...")
    print("-" * 80)

    config = WannSymConfig(
        apply_hermitization=not args.no_hermitize,
        apply_time_reversal=not args.no_time_reversal and has_soc,
        threshold=args.symm_threshold,
        sym_tolerance=args.sym_tolerance,
        verbose=True,
    )

    hr_sym, result = symmetrize_hr(
        hr_data,
        lattice_vectors=lattice_vectors,
        atom_positions_frac=atom_positions_frac,
        atom_numbers=atom_numbers,
        orbital_types_per_atom=wannier_orbital_structure,
        has_soc=has_soc,
        config=config,
    )

    # --- Step 5: Write output ---
    output_file = args.output or f"{hr_file}_nsymm{result.nsymm}"
    nrpt_written = hr_sym.to_file(output_file, threshold=args.symm_threshold)
    result.output_file = output_file

    print()
    print("=" * 80)
    print("STAGE 3 COMPLETE!")
    print("=" * 80)
    print(f"  Output: {output_file}")
    print(f"  R-points: {result.nrpt_original} → {result.nrpt_symmetrized} "
          f"(written: {nrpt_written})")
    print(f"  Symmetry: {result.space_group} ({result.nsymm} operations)")
    print(f"  Max change: {result.max_change:.6e}")
    print()
    print("The symmetrized HR file can be used with wannier_tools or other")
    print("tight-binding post-processing codes.")
    print("=" * 80)


def stage4_plot_bands(args):
    """
    Stage 4: Plot LCAO band structure with PDWF projectability coloring.

    Computes eigenvalues along a high-symmetry k-path and generates a
    two-panel plot with projectability coloring and projected DOS.
    """
    from lcao_wannier.band_plot import (
        run_band_structure, parse_custom_kpath, get_kpath_for_lattice,
        PlotConfig,
    )
    from lcao_wannier.basis_parser import parse_basis_shells, get_atom_list
    from lcao_wannier.valence_config import (
        build_target_mask, compute_num_wann, summarize_config,
    )
    from lcao_wannier.lcao_pdwf import (
        compute_lowdin_projectability, classify_bands, determine_windows,
        ClassificationParams, print_pdwf_summary,
    )
    from lcao_wannier.band_selection import estimate_fermi_energy

    print("=" * 80)
    print("STAGE 4: LCAO Band Structure Plot")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # ---- Parse Crystal23 output ----
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    # Apply user --k-grid override (for memory-limited systems or 2D slabs)
    if getattr(args, 'k_grid', None) is not None:
        original_kgrid = params.k_grid
        params.k_grid = tuple(args.k_grid)
        print(f"  Overriding k-grid: {original_kgrid} -> {params.k_grid} "
              f"(user --k-grid)")
    # Apply user --fermi-energy override (bypass SPINLOCK/level-shift-corrupted value)
    if getattr(args, 'fermi_energy', None) is not None:
        original_fermi = params.fermi_energy
        params.fermi_energy = float(args.fermi_energy)
        print(f"  Overriding Fermi energy: {original_fermi} eV -> "
              f"{params.fermi_energy} eV (user --fermi-energy)")
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    has_soc = params.has_soc
    print(f"  Fermi energy: {params.fermi_energy if params.fermi_energy else 'Not found'}")
    print(f"  K-grid: {params.k_grid}")
    print(f"  SOC: {'Yes' if has_soc else 'No'}")

    # Organize matrices
    H_R_dict = {}
    S_R_dict = {}
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        mat_type = mat_info['type']
        data = mat_info['data']

        if mat_type == 'overlap':
            S_R_dict[R] = data
        elif mat_type == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data

    # Create full matrices
    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, params.num_ao, lattice_vectors_list
        )
    else:
        H_full_list, S_full_list = create_nonsoc_full_matrices(
            H_R_dict, S_R_dict, lattice_vectors_list
        )

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    print(f"  {len(real_space_matrices)} R-vectors")

    # ---- PDWF analysis on uniform grid (unless --no-pdwf) ----
    target_mask = None
    classification = None
    windows = None
    e_fermi = None

    no_pdwf = getattr(args, 'no_pdwf', False)

    if not no_pdwf:
        print("\nStep 2: PDWF analysis on uniform grid...")
        print("-" * 80)

        shells, num_ao = parse_basis_shells(lines, num_atoms=params.num_atoms)
        atoms = get_atom_list(shells)

        kgrid = params.k_grid if params.k_grid else [6, 6, 6]
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=kgrid,
            lattice_vectors=lattice_vectors,
        )
        engine.solve_all_kpoints(parallel=not getattr(args, 'no_parallel', False),
                                 validate_overlap=False)

        nb = engine.num_orbitals
        nk = engine.num_kpoints
        print(f"  {nk} k-points, {nb} bands")

        # Build target mask
        target_mask = build_target_mask(shells, has_soc=has_soc, verbose=False)
        matrix_size = engine.S_k_list[0].shape[0]
        if len(target_mask) != matrix_size:
            if len(target_mask) > matrix_size:
                target_mask = target_mask[:matrix_size]
            else:
                extended = np.zeros(matrix_size, dtype=bool)
                extended[:len(target_mask)] = target_mask
                target_mask = extended

        num_wann = compute_num_wann(atoms, has_soc=has_soc)
        print(f"  num_wann = {num_wann}")
        print(summarize_config(atoms, has_soc=has_soc))

        # Fermi energy
        eigenvalues = np.array(engine.eigenvalues_list)
        if params.fermi_energy is not None:
            e_fermi = params.fermi_energy
        else:
            e_fermi = estimate_fermi_energy(engine.eigenvalues_list, method='midgap')
        print(f"  E_Fermi = {e_fermi:.4f} eV")

        # PDWF classification
        proj_grid = compute_lowdin_projectability(
            engine.eigenvectors_list, engine.S_k_list, target_mask,
        )
        classification = classify_bands(
            proj_grid, eigenvalues, num_wann,
            ClassificationParams(
                p_high=getattr(args, 'pdwf_p_high', 0.95),
                p_low=getattr(args, 'pdwf_p_low', 0.10),
                e_fermi=e_fermi,
            ),
        )
        windows = determine_windows(classification, eigenvalues, e_fermi)
        print_pdwf_summary(classification, windows, eigenvalues, e_fermi, [])
    else:
        print("\nStep 2: Skipping PDWF analysis (--no-pdwf)")
        # Still need Fermi energy
        if params.fermi_energy is not None:
            e_fermi = params.fermi_energy
        else:
            # Quick solve to estimate Fermi energy
            kgrid = params.k_grid if params.k_grid else [6, 6, 6]
            engine = Wannier90Engine(
                real_space_matrices=real_space_matrices,
                k_grid=kgrid,
                lattice_vectors=lattice_vectors,
            )
            engine.solve_all_kpoints(parallel=not getattr(args, 'no_parallel', False),
                                     validate_overlap=False)
            e_fermi = estimate_fermi_energy(engine.eigenvalues_list, method='midgap')
        print(f"  E_Fermi = {e_fermi:.4f} eV")

    # ---- Determine k-path ----
    print("\nStep 3: Band structure computation...")
    print("-" * 80)

    kpath_spec = None
    kpath_type = getattr(args, 'kpath', 'auto')

    if kpath_type == 'custom':
        custom_str = getattr(args, 'custom_kpath', None)
        if custom_str is None:
            print("ERROR: --custom-kpath required with --kpath custom")
            sys.exit(1)
        kpath_spec = parse_custom_kpath(custom_str, npts=getattr(args, 'npts', 60))
    elif kpath_type != 'auto':
        kpath_spec = get_kpath_for_lattice(kpath_type, npts=getattr(args, 'npts', 60))

    # Plot configuration
    energy_range = getattr(args, 'energy_range', None)
    if energy_range is None:
        energy_range = (-20.0, 25.0)
    else:
        energy_range = tuple(energy_range)

    plot_config = PlotConfig(energy_range=energy_range)

    # Output path
    output_plot = getattr(args, 'output_plot', None)
    if output_plot is None:
        output_plot = f"{args.seedname}_bands.png"

    # Run band structure
    band_data = run_band_structure(
        real_space_matrices=real_space_matrices,
        lattice_vectors=lattice_vectors,
        e_fermi=e_fermi,
        output_path=output_plot,
        kpath_spec=kpath_spec,
        npts=getattr(args, 'npts', 60),
        target_mask=target_mask,
        classification=classification,
        windows=windows,
        config=plot_config,
        seedname=args.seedname,
        verbose=True,
    )

    print()
    print("=" * 80)
    print("STAGE 4 COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="LCAO-to-Wannier90 Multi-Stage Workflow Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  Stage 1: Create .win parameter file
    python %(prog)s --stage 1 --input material.out --seedname material

  Then run Wannier90 preprocessing:
    wannier90.x -pp material

  Stage 2: Create data files using .nnkp neighbors
    python %(prog)s --stage 2 --input material.out --seedname material

  Then run Wannier90:
    wannier90.x material

  Stage 3: Symmetrize the tight-binding Hamiltonian
    python %(prog)s --stage 3 --input material.out --seedname material

METHODS (Stages 1-2):
  --method projectability (DEFAULT)
    Select bands by projectability onto LCAO basis.
    Bands with p_avg >= threshold are kept. No energy window needed.
    Tune with --proj-threshold (default: 0.9).

  --method direct
    Use ALL LCAO orbitals as Wannier functions (num_wann = num_basis).
    Skips Wannier90 spread minimization (num_iter = 0).
    Warning issued for num_basis > 200; use --force to override.

  --method symmetry
    Pre-Wannierization symmetry enforcement + irrep band selection.
    Symmetrizes H(R)/S(R), enforces Hermiticity and time-reversal,
    then selects bands using irreducible representation analysis.
    Requires spglib. Tune detection with --sym-tolerance.

STAGE 3 OPTIONS:
  --hr-file PATH      Path to wannier90_hr.dat (default: {seedname}_hr.dat)
  --no-hermitize      Skip Hermitization step
  --no-time-reversal  Skip time-reversal symmetry enforcement
  --symm-threshold    Threshold for dropping small hoppings (default: 1e-9)
  --output PATH       Output filename (default: {hr_file}_nsymm{N})

EXAMPLES:
  # Bismuth with default projectability method
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth
  wannier90.x -pp bismuth
  python %(prog)s --stage 2 --input Bi.out --seedname bismuth
  wannier90.x bismuth

  # Post-Wannierization symmetrization (Stage 3)
  python %(prog)s --stage 3 --input Bi.out --seedname bismuth

  # Direct LCAO mapping (all orbitals)
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --method direct

  # Explicit energy window (overrides projectability)
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --window -6 2
"""
    )

    # Required arguments
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], required=True,
                        help='Stage 1: Create .win | Stage 2: Create .eig/.amn/.mmn | '
                             'Stage 3: Symmetrize wannier90_hr.dat | '
                             'Stage 4: Plot band structure')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CRYSTAL/LCAO output file')
    parser.add_argument('--seedname', '-s', type=str, required=True,
                        help='Seedname for Wannier90 files (e.g., "material")')

    # Optional arguments
    parser.add_argument('--window', type=float, nargs=2, metavar=('E_MIN', 'E_MAX'),
                        help='Energy window in eV relative to Fermi level (default: -5.0 3.0)')
    parser.add_argument('--k-grid', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'),
                        default=None,
                        help='Override Monkhorst-Pack k-grid (default: read from '
                             'CRYSTAL output SHRINK FACT). Use to downsample a '
                             'dense CRYSTAL k-grid for memory-limited systems '
                             '(e.g., --k-grid 6 6 6 for CrI3 on 8-32 GB RAM). '
                             'The k-grid must divide evenly into the R-vector '
                             'set parsed from the LCAO output.')
    parser.add_argument('--fermi-energy', type=float, default=None, metavar='EV',
                        help='Override Fermi energy in eV (bypass CRYSTAL-reported '
                             'value). Required when CRYSTAL SPINLOCK + LEVEL SHIFTER '
                             'corrupts the reported FERMI ENERGY (see "LOCKING - '
                             'FERMI ENERGY ALTERED BY LEVEL SHIFTER" in the LCAO '
                             'output). Estimate from band count: for an insulator, '
                             'set halfway between VBM and CBM; for a metal, use the '
                             'middle of the occupied/empty transition.')
    parser.add_argument('--projections', type=str, nargs='+',
                        help='Wannier90 projection strings (default: random)')
    parser.add_argument('--bands-plot', action='store_true',
                        help='Enable band structure plotting in Wannier90')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel computation')

    # Method selection
    parser.add_argument('--method', type=str,
                        choices=['projectability', 'direct', 'symmetry', 'pdwf', 'auto'],
                        default='projectability',
                        help='Wannierization method (default: projectability). '
                             'pdwf: LCAO-PDWF chemistry-grounded band selection. '
                             'auto: try pdwf, fall back to projectability.')
    parser.add_argument('--proj-threshold', type=float, default=0.9,
                        help='Projectability threshold for band selection '
                             '(default: 0.9, used with --method projectability)')
    parser.add_argument('--num-wann', type=int, default=None,
                        help='Override number of Wannier functions '
                             '(default: auto from frontier detection)')
    parser.add_argument('--force', action='store_true',
                        help='Skip interactive confirmation for large basis sets '
                             '(used with --method direct)')
    parser.add_argument('--sym-tolerance', type=float, default=1e-5,
                        help='Symmetry detection tolerance for spglib '
                             '(default: 1e-5, used with --method symmetry)')
    parser.add_argument('--projection-method', type=str,
                        choices=['weight', 'scdm'],
                        default='weight',
                        help='Projection orbital selection method: '
                             'weight (default, simple ranking) or '
                             'scdm (SCDM-L with QR column pivoting)')

    # PDWF-specific options
    parser.add_argument('--extended', action='store_true',
                        help='Use extended valence config (include semi-core states). '
                             'Used with --method pdwf or --method auto.')
    parser.add_argument('--include-tm-p', action='store_true',
                        help='Include p-channel for transition metals in standard mode. '
                             'Used with --method pdwf or --method auto.')
    parser.add_argument('--pdwf-p-high', type=float, default=0.95,
                        help='PDWF frozen threshold (default: 0.95)')
    parser.add_argument('--pdwf-p-low', type=float, default=0.10,
                        help='PDWF excluded threshold (default: 0.10)')

    # Site symmetry for Wannier90
    parser.add_argument('--site-symmetry', action='store_true',
                        help='Generate .dmn file and enable site_symmetry = .true. '
                             'in Wannier90. Enforces symmetry during Wannierization '
                             '(Sakuma, PRB 87, 235109, 2013). Used in Stage 2.')

    # Symmetry-aware selection for WannSym compatibility
    parser.add_argument('--symmetrize', action='store_true',
                        help='Auto-select num_wann for complete orbital shells '
                             'compatible with WannSym symmetrization (Stage 3). '
                             'Detects space group, analyzes orbital character '
                             'near E_F, and ensures projections form a closed '
                             'representation. Forces SCDM projection method.')
    parser.add_argument('--symm-orbitals', type=str, default='auto',
                        help='Orbital types for --symmetrize: '
                             'auto (default, detect from band structure), '
                             'p, sp, spd, etc.')
    parser.add_argument('--symmetrize-amn', action='store_true',
                        help='Symmetrize AMN projections by averaging over '
                             'equivalent atoms. Gives Wannier90 a symmetric '
                             'starting gauge without affecting other files.')

    # Stage 3 arguments (post-Wannierization symmetrization)
    stage3_group = parser.add_argument_group('Stage 3 options',
                                              'Post-Wannierization symmetrization (WannSym)')
    stage3_group.add_argument('--hr-file', type=str, default=None,
                              help='Path to wannier90_hr.dat for stage 3 '
                                   '(default: {seedname}_hr.dat)')
    stage3_group.add_argument('--output', '-o', type=str, default=None,
                              help='Output filename for symmetrized HR '
                                   '(default: {hr_file}_nsymm{N})')
    stage3_group.add_argument('--no-hermitize', action='store_true',
                              help='Skip Hermitization step (stage 3)')
    stage3_group.add_argument('--no-time-reversal', action='store_true',
                              help='Skip time-reversal symmetry enforcement (stage 3)')
    stage3_group.add_argument('--symm-threshold', type=float, default=1e-9,
                              help='Threshold for dropping small hoppings in '
                                   'symmetrized output (default: 1e-9)')

    # Stage 4 arguments (band structure plot)
    stage4_group = parser.add_argument_group('Stage 4 options',
                                              'Band structure plotting')
    stage4_group.add_argument('--kpath', type=str,
                              choices=['auto', 'hexagonal_2d', 'hexagonal_3d',
                                       'fcc', 'bcc', 'sc', 'custom'],
                              default='auto',
                              help='K-path type (default: auto-detect from lattice)')
    stage4_group.add_argument('--npts', type=int, default=60,
                              help='Number of k-points per segment (default: 60)')
    stage4_group.add_argument('--energy-range', type=float, nargs=2,
                              metavar=('E_MIN', 'E_MAX'), default=None,
                              help='Energy range relative to E_F in eV '
                                   '(default: -20 25)')
    stage4_group.add_argument('--output-plot', type=str, default=None,
                              help='Output plot filename '
                                   '(default: {seedname}_bands.png)')
    stage4_group.add_argument('--no-pdwf', action='store_true',
                              help='Skip PDWF projectability coloring '
                                   '(uniform color bands)')
    stage4_group.add_argument('--custom-kpath', type=str, default=None,
                              help='Custom k-path string: '
                                   '"G:0,0,0;M:0.5,0,0;K:0.333,0.333,0;G:0,0,0"')

    args = parser.parse_args()

    # Execute appropriate stage
    if args.stage == 1:
        stage1_create_win(args)
    elif args.stage == 2:
        stage2_create_data_files(args)
    elif args.stage == 3:
        stage3_symmetrize_hr(args)
    elif args.stage == 4:
        stage4_plot_bands(args)


if __name__ == '__main__':
    main()
