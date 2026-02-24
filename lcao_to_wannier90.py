#!/usr/bin/env python3
"""
LCAO-to-Wannier90 Two-Stage Workflow Script

This script implements the correct Wannier90 workflow with two stages:

Stage 1: Generate .win file from CRYSTAL/LCAO output
Stage 2: Generate .eig, .amn, .mmn files using .nnkp neighbor information

Usage:
    # Stage 1: Create .win file
    python lcao_to_wannier90.py --stage 1 --input material.out --seedname material

    # Then run: wannier90.x -pp material

    # Stage 2: Create data files using .nnkp
    python lcao_to_wannier90.py --stage 2 --input material.out --seedname material

    # Then run: wannier90.x material

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
    prepare_real_space_matrices,
    Wannier90Engine,
    suggest_optimal_window,
    analyze_band_window,
    parse_atoms_from_crystal_output,
    estimate_fermi_energy,
)
from lcao_wannier.projectability import select_bands_by_projectability, smart_select_bands

# Threshold above which --method direct warns the user
LARGE_BASIS_THRESHOLD = 200


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
        engine._num_bands_for_win = len(result.selected_band_indices)
        engine._dis_win = result.recommended_dis_win
        engine._dis_froz = result.recommended_dis_froz

        if result.recommended_dis_win is not None:
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
    print("\nSelecting projection orbitals...")
    print("-" * 80)
    engine.select_projections(verbose=True)


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


def _infer_projections(atoms, num_wann, has_soc):
    """Infer Wannier90 projection strings from atoms and num_wann.

    Tries to match num_wann to standard orbital sets (per atom × spinor_factor):
      - s:     1 orbital per atom
      - p:     3 orbitals per atom
      - sp3:   4 orbitals per atom
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

    # Try s+p (sp3): 4 per atom
    if orb_per_atom == 4:
        projections = [f"{e}:sp3" for e in elements]
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
    """Apply window-based band selection (fallback when --window is explicit)."""
    e_min, e_max = args.window

    print("\nWindow-based band selection...")
    print("-" * 80)

    result = analyze_band_window(
        engine.eigenvalues_list,
        outer_window=(e_min, e_max),
        e_fermi=engine.e_fermi,
        window_is_relative=True
    )

    num_frozen = result.num_wann
    if num_frozen == 0:
        print("ERROR: No bands found in the energy window!")
        print("Please adjust the energy window and try again.")
        sys.exit(1)

    engine.num_wann = num_frozen
    engine.selected_band_indices = result.frozen_indices

    print(f"Selected {num_frozen} bands for Wannier functions")
    print(f"  Energy range: [{result.frozen_energy_range[0]:.2f}, {result.frozen_energy_range[1]:.2f}] eV")

    # Select projection orbitals
    print("\nSelecting projection orbitals...")
    print("-" * 80)
    engine.select_projections(verbose=True)


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
        # For non-SOC, create the (R_cartesian, matrix) list format manually
        H_full_list = []
        S_full_list = []

        for R_int, H_mats in H_R_dict.items():
            # Convert integer R to Cartesian
            R_cart = lattice_vectors.T @ np.array(R_int)
            # Use spin channel 0 (or average if multiple)
            H_mat = list(H_mats.values())[0] if isinstance(H_mats, dict) else H_mats
            H_full_list.append((R_cart, H_mat))

        for R_int, S_mat in S_R_dict.items():
            R_cart = lattice_vectors.T @ np.array(R_int)
            S_full_list.append((R_cart, S_mat))

        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} matrices (no SOC)")

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

    if args.method == 'symmetry':
        print("ERROR: Symmetry-indicator method is not yet implemented.")
        print("Please use --method projectability (default) or --method direct.")
        raise NotImplementedError(
            "Method 'symmetry' (symmetry-indicator approach) is planned for "
            "a future release. Use --method projectability or --method direct."
        )
    elif args.method == 'direct':
        _apply_method_direct(engine, args, has_soc, params)
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

    # Auto-detect kpoint path for band structure plots
    kpoint_path = None
    if args.bands_plot:
        from lcao_wannier.win_file import KPATH_HEXAGONAL_2D
        # Detect 2D hexagonal system: a3 >> a1, a2
        a1_len = np.linalg.norm(engine.lattice_vectors[0])
        a3_len = np.linalg.norm(engine.lattice_vectors[2])
        if a3_len > 10 * a1_len:
            kpoint_path = KPATH_HEXAGONAL_2D
            print(f"  Auto-detected 2D hexagonal lattice → M-Γ-K band path")

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
        # For non-SOC, create the (R_cartesian, matrix) list format manually
        H_full_list = []
        S_full_list = []

        for R_int, H_mats in H_R_dict.items():
            # Convert integer R to Cartesian
            R_cart = lattice_vectors.T @ np.array(R_int)
            # Use spin channel 0 (or average if multiple)
            H_mat = list(H_mats.values())[0] if isinstance(H_mats, dict) else H_mats
            H_full_list.append((R_cart, H_mat))

        for R_int, S_mat in S_R_dict.items():
            R_cart = lattice_vectors.T @ np.array(R_int)
            S_full_list.append((R_cart, S_mat))

        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} matrices (no SOC)")

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

    # Solve eigenvalue problems
    print("\nStep 7: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel)
    print("✓ Eigenvalue problems solved")

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

    if args.method == 'symmetry':
        print("ERROR: Symmetry-indicator method is not yet implemented.")
        print("Please use --method projectability (default) or --method direct.")
        raise NotImplementedError(
            "Method 'symmetry' (symmetry-indicator approach) is planned for "
            "a future release. Use --method projectability or --method direct."
        )
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

    # Write only the data files, not .win (already exists)
    engine.write_files(
        verbose=True,
        write_win=False,  # Don't overwrite .win
        use_nnkp=True,    # Use .nnkp neighbors (CRITICAL!)
    )

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.eig")
    print(f"✓ Created: {args.seedname}.amn")
    print(f"✓ Created: {args.seedname}.mmn")
    print()
    print("All files generated with correct neighbor structure from .nnkp!")
    print()
    print("NEXT STEP:")
    print(f"  Run Wannier90 to generate maximally localized Wannier functions:")
    print(f"  → wannier90.x {args.seedname}")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="LCAO-to-Wannier90 Two-Stage Workflow Script",
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

METHODS:
  --method projectability (DEFAULT)
    Select bands by projectability onto LCAO basis.
    Bands with p_avg >= threshold are kept. No energy window needed.
    Tune with --proj-threshold (default: 0.9).

  --method direct
    Use ALL LCAO orbitals as Wannier functions (num_wann = num_basis).
    Skips Wannier90 spread minimization (num_iter = 0).
    Warning issued for num_basis > 200; use --force to override.

  --method symmetry
    (NOT YET IMPLEMENTED) Group-theory based projection selection.

EXAMPLES:
  # Bismuth with default projectability method
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth
  wannier90.x -pp bismuth
  python %(prog)s --stage 2 --input Bi.out --seedname bismuth
  wannier90.x bismuth

  # Direct LCAO mapping (all orbitals)
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --method direct
  wannier90.x -pp bismuth
  python %(prog)s --stage 2 --input Bi.out --seedname bismuth --method direct

  # Explicit energy window (overrides projectability)
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --window -6 2
"""
    )

    # Required arguments
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                        help='Stage 1: Create .win | Stage 2: Create .eig/.amn/.mmn')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CRYSTAL/LCAO output file')
    parser.add_argument('--seedname', '-s', type=str, required=True,
                        help='Seedname for Wannier90 files (e.g., "material")')

    # Optional arguments
    parser.add_argument('--window', type=float, nargs=2, metavar=('E_MIN', 'E_MAX'),
                        help='Energy window in eV relative to Fermi level (default: -5.0 3.0)')
    parser.add_argument('--projections', type=str, nargs='+',
                        help='Wannier90 projection strings (default: random)')
    parser.add_argument('--bands-plot', action='store_true',
                        help='Enable band structure plotting in Wannier90')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel computation')

    # Method selection
    parser.add_argument('--method', type=str,
                        choices=['projectability', 'direct', 'symmetry'],
                        default='projectability',
                        help='Wannierization method (default: projectability)')
    parser.add_argument('--proj-threshold', type=float, default=0.9,
                        help='Projectability threshold for band selection '
                             '(default: 0.9, used with --method projectability)')
    parser.add_argument('--num-wann', type=int, default=None,
                        help='Override number of Wannier functions '
                             '(default: auto from frontier detection)')
    parser.add_argument('--force', action='store_true',
                        help='Skip interactive confirmation for large basis sets '
                             '(used with --method direct)')

    args = parser.parse_args()

    # Execute appropriate stage
    if args.stage == 1:
        stage1_create_win(args)
    elif args.stage == 2:
        stage2_create_data_files(args)


if __name__ == '__main__':
    main()
