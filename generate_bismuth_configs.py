#!/usr/bin/env python3
"""
Generate Wannier90 input files for Bismuth bilayer (SOC) using
Lowdin PDWF and midpoint+unitarize methods.

Configs:
  pdwf_lowdin_berry/    - PDWF AMN, Lowdin MMN with Berry phase
  pdwf_lowdin_no_berry/ - PDWF AMN, Lowdin MMN without Berry phase
  pdwf_unitarize/       - PDWF AMN, midpoint MMN with SVD unitarization

Eigensolve is done once; only MMN method differs between configs.
"""

import os
import sys
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lcao_wannier.parser import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    parse_atomic_basis_info,
    create_spin_block_matrices,
)
from lcao_wannier.engine import Wannier90Engine
from lcao_wannier.wannier90 import (
    write_eig_file,
    write_amn_file_pdwf,
    write_mmn_file_lcao,
)
from lcao_wannier.win_file import (
    write_win_file,
    create_win_config_from_engine,
    parse_atoms_from_crystal_output,
)
from lcao_wannier.kpoints import convert_neighbor_list_to_dict_format, read_nnkp_neighbors
from lcao_wannier.basis_parser import parse_basis_shells, get_atom_list
from lcao_wannier.valence_config import build_target_mask, compute_num_wann
from lcao_wannier.lcao_pdwf import (
    compute_lowdin_projectability,
    classify_bands,
    determine_windows,
    ClassificationParams,
)

CRYSTAL_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tests", "Bismuth_basis_40.out"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Bismuth"
)
WANNIER90_X = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external", "wannier90-3.1.0", "wannier90.x"
)


def prepare_real_space_matrices(H_list, S_list, lattice_vectors):
    """Convert list-of-tuples format to dict format keyed by integer R."""
    rsm = {}
    # Build S lookup by Cartesian R
    S_by_R = {}
    for R_cart, S in S_list:
        key = tuple(np.round(R_cart, 8))
        S_by_R[key] = S

    for R_cart, H in H_list:
        key = tuple(np.round(R_cart, 8))
        if key in S_by_R:
            R_int = tuple(np.round(
                np.linalg.solve(lattice_vectors.T, R_cart)
            ).astype(int))
            rsm[R_int] = {'H': H, 'S': S_by_R[key]}
    return rsm


def main():
    # ── Parse Crystal output ──
    print("Parsing Crystal output...")
    with open(CRYSTAL_OUT) as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"  E_F = {params.fermi_energy:.6f} eV")
    print(f"  num_ao = {params.num_ao} (per spin)")
    print(f"  num_atoms = {params.num_atoms}")

    # Organize matrices by R-vector
    H_R_dict = {}
    S_R_dict = {}
    num_basis = None
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        mat_type = mat_info['type']
        data = mat_info['data']
        if data is None:
            continue
        if mat_type == 'overlap':
            S_R_dict[R] = data
            if num_basis is None:
                num_basis = data.shape[0]
        elif mat_type == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data
            if num_basis is None:
                num_basis = data.shape[0]

    print(f"  Basis per spin: {num_basis}")

    # ── Create SOC spin-block matrices ──
    print("Creating SOC spin-block matrices...")
    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict, S_R_dict, num_basis, lattice_vectors_list
    )
    num_orbitals = H_full_list[0][1].shape[0]
    print(f"  SOC matrix dimension: {num_orbitals} x {num_orbitals}")

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    print(f"  R-vectors = {len(real_space_matrices)}")

    # ── Parse basis for PDWF ──
    shells, num_ao_spatial = parse_basis_shells(lines, num_atoms=params.num_atoms)
    atoms_list = get_atom_list(shells)
    # SOC system: has_soc=True doubles the Wannier function count
    target_mask = build_target_mask(
        shells, extended=False, include_tm_p=False, has_soc=True
    )
    num_wann = compute_num_wann(
        atoms_list, extended=False, include_tm_p=False, has_soc=True
    )
    print(f"  Target AOs: {int(np.sum(target_mask))} / {len(target_mask)}")
    print(f"  num_wann: {num_wann}")

    # ── Initialize engine ──
    e_fermi = params.fermi_energy

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        outer_window=(-18.0, 12.0),
        e_fermi=e_fermi,
        num_electrons=params.num_electrons,
        seedname="bismuth",
    )

    # Set atomic info (doubled for SOC)
    atomic_info = parse_atomic_basis_info(lines)
    engine.atom_positions = atomic_info.atom_positions
    if num_orbitals == 2 * atomic_info.num_basis:
        engine.basis_atom_map = np.concatenate([
            atomic_info.basis_atom_map,
            atomic_info.basis_atom_map,
        ])
        print(f"  SOC: doubled basis_atom_map ({len(engine.basis_atom_map)} orbitals)")
    else:
        engine.basis_atom_map = atomic_info.basis_atom_map

    print("Solving eigenvalues...")
    engine.solve_all_kpoints(parallel=True)
    print(f"  Solved {engine.num_kpoints} k-points")

    atoms_pos, _ = parse_atoms_from_crystal_output(lines)

    # ── Band classification ──
    eigenvalues = np.array(engine.eigenvalues_list)

    # Try PDWF projectability classification first
    proj = compute_lowdin_projectability(
        engine.eigenvectors_list, engine.S_k_list, target_mask,
    )

    classification = classify_bands(
        proj, eigenvalues, num_wann,
        ClassificationParams(p_high=0.90, p_low=0.10, e_fermi=e_fermi),
    )
    windows = determine_windows(classification, eigenvalues, e_fermi)

    # Energy-range freezing when no frozen bands from projectability
    if len(classification.frozen_indices) == 0:
        avg_e = classification.band_energies
        avg_p = classification.avg_projectability
        for b in range(len(avg_p)):
            if (avg_p[b] >= 0.10 and
                avg_e[b] <= e_fermi + 3.0 and
                classification.category[b] != 'excluded'):
                classification.category[b] = 'frozen'
        classification.frozen_indices = np.where(
            classification.category == 'frozen'
        )[0]
        classification.disent_indices = np.where(
            classification.category == 'disent'
        )[0]
        windows = determine_windows(classification, eigenvalues, e_fermi)

    all_active = np.union1d(
        classification.frozen_indices, classification.disent_indices
    )

    # Fallback: if classification still gave nothing, use energy window
    # Must include 6s bands (~-14 eV below E_F) for sp3 Wannier functions
    if len(all_active) == 0:
        print("  PDWF classification found no active bands, using energy window fallback")
        # Select bands whose mean energy falls within a wide window around E_F
        # The 6s bands at -14 to -10 eV are essential for sp3 WFs
        nb = eigenvalues.shape[1]
        mean_eigs = np.mean(eigenvalues, axis=0)
        win_lo = e_fermi - 18.0
        win_hi = e_fermi + 12.0
        in_window = np.where(
            (mean_eigs >= win_lo) & (mean_eigs <= win_hi)
        )[0]
        all_active = in_window
        # Set windows matching benchmark
        active_eigs = eigenvalues[:, all_active]
        windows.dis_win_min = float(np.min(active_eigs)) - 2.0
        windows.dis_win_max = float(np.max(active_eigs)) + 2.0
        windows.dis_froz_min = e_fermi - 5.0
        windows.dis_froz_max = e_fermi + 2.0

    # Expand to ratio >= 1.2 if needed
    min_bands = max(int(np.ceil(num_wann * 1.2)), num_wann + 1)
    if len(all_active) < min_bands:
        nb = eigenvalues.shape[1]
        candidates = []
        active_set = set(all_active)
        hi = windows.dis_win_max + 10.0 if windows.dis_win_max else (
            np.max(eigenvalues[:, all_active]) + 12.0
        )
        lo = np.min(eigenvalues[:, all_active]) - 2.0
        for b in range(nb):
            if b in active_set:
                continue
            band_eigs = eigenvalues[:, b]
            if np.any((band_eigs >= lo) & (band_eigs <= hi)):
                candidates.append((np.mean(band_eigs), b))
        candidates.sort()
        for _, b in candidates:
            all_active = np.union1d(all_active, [b])
            if len(all_active) >= min_bands:
                break
        active_eigs = eigenvalues[:, all_active]
        windows.dis_win_min = float(np.min(active_eigs)) - 2.0
        windows.dis_win_max = float(np.max(active_eigs)) + 2.0

    band_indices = all_active
    print(f"\n  Selected bands: {len(band_indices)} "
          f"(indices {band_indices[0]}-{band_indices[-1]})")
    print(f"  Ratio: {len(band_indices)/num_wann:.2f}")
    if windows.dis_froz_min is not None:
        print(f"  Frozen: [{windows.dis_froz_min - e_fermi:+.1f}, "
              f"{windows.dis_froz_max - e_fermi:+.1f}] eV rel E_F")
    if windows.dis_win_min is not None:
        print(f"  Outer:  [{windows.dis_win_min - e_fermi:+.1f}, "
              f"{windows.dis_win_max - e_fermi:+.1f}] eV rel E_F")

    # ── Set engine state ──
    engine.num_wann = num_wann
    engine.selected_band_indices = band_indices
    engine._num_bands_for_win = len(band_indices)
    if windows.dis_win_min is not None and len(band_indices) > num_wann:
        engine._dis_win = (
            windows.dis_win_min - e_fermi,
            windows.dis_win_max - e_fermi,
        )
    if windows.dis_froz_min is not None:
        engine._dis_froz = (
            windows.dis_froz_min - e_fermi,
            windows.dis_froz_max - e_fermi,
        )

    # ── Prepare selected eigenvectors (for midpoint method) ──
    eigenvectors_selected = [
        engine.eigenvectors_list[ik][:, band_indices]
        for ik in range(engine.num_kpoints)
    ]

    # ── Convert neighbor list ──
    neighbor_list = engine.neighbor_list
    if neighbor_list and isinstance(neighbor_list[0][0], tuple):
        neighbor_list = convert_neighbor_list_to_dict_format(
            neighbor_list, engine.recip_lattice, kpoints=engine.kpoints
        )

    # ── Projections for .win ──
    projections = ["Bi:l=0;l=1"]

    # ── Generate configs ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = [
        {"name": "pdwf_lowdin_berry",    "mmn": "lowdin",          "unitarize": False},
        {"name": "pdwf_lowdin_no_berry", "mmn": "lowdin_no_berry", "unitarize": False},
        {"name": "pdwf_unitarize",       "mmn": "midpoint",        "unitarize": True},
        {"name": "pdwf_no_unitarize",    "mmn": "midpoint",        "unitarize": False},
    ]

    for cfg in configs:
        name = cfg["name"]
        outdir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(outdir, exist_ok=True)
        seedpath = os.path.join(outdir, "bismuth")

        print(f"\n{'='*70}")
        print(f"Config: {name}")
        print(f"  MMN: {cfg['mmn']}, Unitarize: {cfg['unitarize']}")
        print(f"  Output: {outdir}/")
        print(f"{'='*70}")

        # Step 1: Write .win FIRST
        old_seedname = engine.seedname
        engine.seedname = seedpath
        win_config = create_win_config_from_engine(
            engine,
            atoms=atoms_pos,
            projections=projections,
            spinors=True,       # SOC system
            bands_plot=False,
            write_hr=True,
            use_bloch_phases=False,
            num_iter=5000,
            dis_num_iter=1000,
        )
        write_win_file(
            os.path.join(outdir, "bismuth"), win_config, verbose=False
        )
        engine.seedname = old_seedname
        print(f"  -> bismuth.win")

        # Step 2: Run wannier90.x -pp to generate .nnkp
        print(f"  Running wannier90.x -pp ...")
        result = subprocess.run(
            [WANNIER90_X, "-pp", "bismuth"],
            cwd=outdir, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: wannier90.x -pp failed:\n{result.stderr}")
            continue
        nnkp_file = f"{seedpath}.nnkp"
        if not os.path.exists(nnkp_file):
            print(f"  ERROR: .nnkp not generated")
            continue
        print(f"  -> bismuth.nnkp")

        # Step 3: Read neighbor list from .nnkp
        nnkp_neighbors = read_nnkp_neighbors(
            nnkp_file,
            recip_lattice=engine.recip_lattice,
            kpoints=engine.kpoints,
        )
        num_nnkp_neighbors = len(nnkp_neighbors[0])
        print(f"  Read {num_nnkp_neighbors} neighbors/k-point from .nnkp")

        # Step 4: Write .eig
        write_eig_file(
            f"{seedpath}.eig",
            [engine.eigenvalues_list[ik][band_indices] - e_fermi
             for ik in range(engine.num_kpoints)],
            engine.num_kpoints,
            len(band_indices),
        )
        print(f"  -> bismuth.eig")

        # Step 5: Write .amn (PDWF for all configs)
        write_amn_file_pdwf(
            f"{seedpath}.amn",
            engine.eigenvectors_list,
            engine.S_k_list,
            target_mask,
            band_indices,
            engine.num_kpoints,
            num_wann,
        )
        print(f"  -> bismuth.amn (PDWF SVD, "
              f"{int(np.sum(target_mask))} target AOs -> {num_wann} WFs)")

        # Step 6: Write .mmn using the .nnkp neighbor list
        mmn_kwargs = dict(
            convention='pi',
            use_direct_method=True,
            verbose=False,
            stacked=engine._stacked,
            unitarize=cfg["unitarize"],
            method=cfg["mmn"],
        )
        if cfg["mmn"] in ("lowdin", "lowdin_no_berry"):
            write_mmn_file_lcao(
                f"{seedpath}.mmn",
                engine.eigenvectors_list,
                engine.kpoints,
                engine.real_space_matrices,
                engine.lattice_vectors,
                nnkp_neighbors,
                engine.atom_positions,
                engine.basis_atom_map,
                engine.num_kpoints,
                len(band_indices),
                S_k_list=engine.S_k_list,
                band_indices=band_indices,
                recip_lattice=engine.recip_lattice,
                **mmn_kwargs,
            )
        else:
            write_mmn_file_lcao(
                f"{seedpath}.mmn",
                eigenvectors_selected,
                engine.kpoints,
                engine.real_space_matrices,
                engine.lattice_vectors,
                nnkp_neighbors,
                engine.atom_positions,
                engine.basis_atom_map,
                engine.num_kpoints,
                len(band_indices),
                **mmn_kwargs,
            )
        tag = f"{cfg['mmn']}" + (" +unitarize" if cfg["unitarize"] else "")
        print(f"  -> bismuth.mmn ({tag}, {num_nnkp_neighbors} neighbors from .nnkp)")

    print(f"\n{'='*70}")
    print("All configs generated!")
    print(f"{'='*70}")
    print("\nTo run on cluster:")
    print("  cd <config_dir>")
    print("  wannier90.x bismuth")
    print()

    for cfg in configs:
        d = os.path.join(OUTPUT_DIR, cfg["name"])
        files = sorted([f for f in os.listdir(d) if f.startswith("bismuth.")])
        print(f"  {cfg['name']}/: {', '.join(files)}")


if __name__ == '__main__':
    main()
