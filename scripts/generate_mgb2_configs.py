#!/usr/bin/env python3
"""
Generate multiple Wannier90 input file sets for MgB2.

8 configs: 2 AMN methods x 4 MMN approaches:
  AMN: scdm (SCDM orbital selection), pdwf (PDWF SVD)
  MMN: none (raw midpoint), svd (polar decomposition), soft (tanh knee), lowdin (Löwdin method)

All configs now use corrected integer R-vectors in the Fourier transform.
Eigensolve is done once; only AMN/MMN differ between configs.
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
    create_nonsoc_full_matrices,
)
from lcao_wannier.engine import Wannier90Engine
from lcao_wannier.wannier90 import (
    write_eig_file,
    write_amn_file_lcao,
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

CRYSTAL_OUT = "/Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier/MgB2/MgB2_basis_141.out"
OUTPUT_DIR = "/Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier/MgB2"
WANNIER90_X = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external", "wannier90-3.1.0", "wannier90.x"
)


def prepare_real_space_matrices(H_list, S_list, lattice_vectors):
    """Convert list-of-tuples format to dict format keyed by integer R.

    The parser returns (R_cartesian, matrix) pairs. The Fourier transform
    expects integer lattice vector coefficients (n1, n2, n3) as keys,
    computing phase = exp(2πi · k_frac · R_int).
    """
    rsm = {}
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
    print(f"  num_ao = {params.num_ao}")

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

    H_full_list, S_full_list = create_nonsoc_full_matrices(
        H_R_dict, S_R_dict, lattice_vectors_list
    )
    real_space_matrices = prepare_real_space_matrices(H_full_list, S_full_list, lattice_vectors)
    print(f"  R-vectors = {len(real_space_matrices)}")

    # ── Parse basis for PDWF ──
    shells, num_ao_spatial = parse_basis_shells(lines, num_atoms=params.num_atoms)
    atoms_list = get_atom_list(shells)
    target_mask = build_target_mask(shells, extended=False, include_tm_p=False, has_soc=False)
    num_wann = compute_num_wann(atoms_list, extended=False, include_tm_p=False, has_soc=False)
    print(f"  Target AOs: {int(np.sum(target_mask))} / {len(target_mask)}")
    print(f"  num_wann: {num_wann}")

    # ── Initialize engine and solve eigenvalues (once) ──
    e_fermi = params.fermi_energy

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        outer_window=(-5.0, 3.0),
        e_fermi=e_fermi,
        num_electrons=params.num_electrons,
        seedname="mgb2",
    )

    atomic_info = parse_atomic_basis_info(lines)
    engine.atom_positions = atomic_info.atom_positions
    engine.basis_atom_map = atomic_info.basis_atom_map

    print("Solving eigenvalues...")
    engine.solve_all_kpoints(parallel=True)
    print(f"  Solved {engine.num_kpoints} k-points")

    atoms_pos, _ = parse_atoms_from_crystal_output(lines)

    # ── Band classification ──
    eigenvalues = np.array(engine.eigenvalues_list)
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
        classification.frozen_indices = np.where(classification.category == 'frozen')[0]
        classification.disent_indices = np.where(classification.category == 'disent')[0]
        windows = determine_windows(classification, eigenvalues, e_fermi)

    all_active = np.union1d(classification.frozen_indices, classification.disent_indices)

    # Expand to ratio >= 1.5
    min_bands = max(int(np.ceil(num_wann * 1.5)), num_wann + 1)
    if len(all_active) < min_bands:
        nb = eigenvalues.shape[1]
        candidates = []
        for b in range(nb):
            if b in set(all_active):
                continue
            band_eigs = eigenvalues[:, b]
            hi = (windows.dis_win_max + 10.0) if windows.dis_win_max else (np.max(eigenvalues[:, all_active]) + 12.0)
            lo = np.min(eigenvalues[:, all_active]) - 2.0
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
    print(f"\n  Selected bands: {len(band_indices)} (indices {band_indices[0]}-{band_indices[-1]})")
    print(f"  Ratio: {len(band_indices)/num_wann:.2f}")
    if windows.dis_froz_min is not None:
        print(f"  Frozen: [{windows.dis_froz_min - e_fermi:+.1f}, {windows.dis_froz_max - e_fermi:+.1f}] eV rel E_F")
    if windows.dis_win_min is not None:
        print(f"  Outer:  [{windows.dis_win_min - e_fermi:+.1f}, {windows.dis_win_max - e_fermi:+.1f}] eV rel E_F")

    # ── Set engine state ──
    engine.num_wann = num_wann
    engine.selected_band_indices = band_indices
    engine._num_bands_for_win = len(band_indices)
    if windows.dis_win_min is not None and len(band_indices) > num_wann:
        engine._dis_win = (windows.dis_win_min - e_fermi, windows.dis_win_max - e_fermi)
    if windows.dis_froz_min is not None:
        engine._dis_froz = (windows.dis_froz_min - e_fermi, windows.dis_froz_max - e_fermi)

    # ── SCDM projection selection ──
    print("\nRunning SCDM projection selection...")
    engine.select_projections(verbose=True, method='scdm')
    scdm_orbital_indices = engine.selected_orbital_indices.copy()
    print(f"  SCDM orbitals: {scdm_orbital_indices}")

    # ── Prepare selected eigenvectors ──
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

    # ── Cosmetic projections for .win ──
    elements = list(dict.fromkeys(sym for sym, _ in atoms_pos))
    projections = [f"{e}:l=0;l=1" for e in elements]

    # ── Generate configs ──
    # 2 AMN methods x 4 MMN conditioning approaches = 8 configs
    # AMN: scdm (SCDM orbital selection), pdwf (PDWF SVD projection)
    # MMN conditioning:
    #   none:   midpoint, raw (no conditioning)
    #   svd:    midpoint + SVD polar decomposition (hard unitarization)
    #   soft:   midpoint + soft-knee tanh conditioning (knee=0.5)
    #   lowdin: Löwdin method (atom-centered Berry phase, different MMN algorithm)
    configs = [
        {"name": "scdm_none",    "amn": "scdm", "mmn": "midpoint", "conditioning": "none"},
        {"name": "scdm_svd",     "amn": "scdm", "mmn": "midpoint", "conditioning": "svd"},
        {"name": "scdm_soft",    "amn": "scdm", "mmn": "midpoint", "conditioning": "soft"},
        {"name": "scdm_lowdin",  "amn": "scdm", "mmn": "lowdin",   "conditioning": "none"},
        {"name": "pdwf_none",    "amn": "pdwf", "mmn": "midpoint", "conditioning": "none"},
        {"name": "pdwf_svd",     "amn": "pdwf", "mmn": "midpoint", "conditioning": "svd"},
        {"name": "pdwf_soft",    "amn": "pdwf", "mmn": "midpoint", "conditioning": "soft"},
        {"name": "pdwf_lowdin",  "amn": "pdwf", "mmn": "lowdin",   "conditioning": "none"},
    ]

    for cfg in configs:
        name = cfg["name"]
        outdir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(outdir, exist_ok=True)
        seedpath = os.path.join(outdir, "mgb2")

        print(f"\n{'='*70}")
        print(f"Config: {name}")
        print(f"  AMN: {cfg['amn']}, MMN: {cfg['mmn']}, Conditioning: {cfg['conditioning']}")
        print(f"  Output: {outdir}/")
        print(f"{'='*70}")

        # Step 1: Write .win FIRST (needed for wannier90.x -pp)
        old_seedname = engine.seedname
        engine.seedname = seedpath
        win_config = create_win_config_from_engine(
            engine,
            atoms=atoms_pos,
            projections=projections,
            spinors=False,
            bands_plot=False,
            write_hr=True,
            use_bloch_phases=False,
            num_iter=2000,
            dis_num_iter=200,
        )
        write_win_file(os.path.join(outdir, "mgb2"), win_config, verbose=False)
        engine.seedname = old_seedname
        print(f"  -> mgb2.win")

        # Step 2: Run wannier90.x -pp to generate .nnkp
        print(f"  Running wannier90.x -pp ...")
        result = subprocess.run(
            [WANNIER90_X, "-pp", "mgb2"],
            cwd=outdir, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: wannier90.x -pp failed:\n{result.stderr}")
            continue
        nnkp_file = f"{seedpath}.nnkp"
        if not os.path.exists(nnkp_file):
            print(f"  ERROR: .nnkp not generated")
            continue
        print(f"  -> mgb2.nnkp")

        # Step 3: Read neighbor list from .nnkp (Wannier90's actual neighbor list)
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
            [engine.eigenvalues_list[ik][band_indices] - e_fermi for ik in range(engine.num_kpoints)],
            engine.num_kpoints,
            len(band_indices),
        )
        print(f"  -> mgb2.eig")

        # Step 5: Write .amn
        if cfg["amn"] == "pdwf":
            write_amn_file_pdwf(
                f"{seedpath}.amn",
                engine.eigenvectors_list,
                engine.S_k_list,
                target_mask,
                band_indices,
                engine.num_kpoints,
                num_wann,
            )
            print(f"  -> mgb2.amn (PDWF SVD, {int(np.sum(target_mask))} target AOs -> {num_wann} WFs)")
        else:
            write_amn_file_lcao(
                f"{seedpath}.amn",
                engine.eigenvectors_list,
                engine.S_k_list,
                scdm_orbital_indices,
                band_indices,
                engine.num_kpoints,
                len(band_indices),
            )
            print(f"  -> mgb2.amn (SCDM, orbitals {scdm_orbital_indices})")

        # Step 6: Write .mmn using the .nnkp neighbor list
        mmn_kwargs = dict(
            convention='pi',
            use_direct_method=True,
            verbose=False,
            stacked=engine._stacked,
            method=cfg["mmn"],
            conditioning=cfg["conditioning"],
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
        tag = f"{cfg['mmn']} cond={cfg['conditioning']}"
        print(f"  -> mgb2.mmn ({tag}, {num_nnkp_neighbors} neighbors from .nnkp)")

    print(f"\n{'='*70}")
    print("All configs generated!")
    print(f"{'='*70}")
    print("\nTo run on cluster, for each config directory:")
    print("  cd <config_dir>")
    print("  wannier90.x mgb2")
    print()

    for cfg in configs:
        d = os.path.join(OUTPUT_DIR, cfg["name"])
        files = sorted([f for f in os.listdir(d) if f.startswith("mgb2.")])
        print(f"  {cfg['name']}/: {', '.join(files)}")


if __name__ == '__main__':
    main()
