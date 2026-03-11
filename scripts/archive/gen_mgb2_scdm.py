#!/usr/bin/env python3
"""Generate pure SCDM Wannier90 input files for MgB2.

No physics-based projections, no orbital mask — just SCDM column pivoting
on the full LCAO basis. Uses 'random' projections in the .win file so
Wannier90 relies entirely on the SCDM-computed .amn file.

Generates multiple num_wann values: 5, 9, 12 (matching scored configs for
comparison).
"""

import sys, os, subprocess, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lcao_wannier.parser import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    parse_atomic_basis_info,
)
from lcao_wannier.engine import Wannier90Engine
from lcao_wannier.utils import prepare_real_space_matrices
from lcao_wannier.win_file import (
    parse_atoms_from_crystal_output,
    create_win_config_from_engine,
    write_win_file,
)
from lcao_to_wannier90 import _build_nonsoc_matrix_lists

INPUT_FILE = os.path.join(ROOT, 'MgB2/MgB2_basis_121.out')
BASE_DIR = os.path.join(ROOT, 'MgB2')
W90_BIN = os.path.abspath(os.path.join(ROOT, 'external/wannier90-3.1.0/wannier90.x'))

# num_wann values to generate (match scored configs for comparison)
SCDM_CONFIGS = [5, 9, 12]


def main():
    t0 = time.time()

    # ================================================================
    # Parse + solve eigenvalues (same as scored configs)
    # ================================================================
    print("=" * 70)
    print("SCDM-only: Parsing + Eigenvalue Solve")
    print("=" * 70)

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)
    has_soc = params.has_soc

    # Build matrices
    H_R_dict, S_R_dict = {}, {}
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        if mat_info['type'] == 'overlap':
            S_R_dict[R] = mat_info['data']
        elif mat_info['type'] == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = mat_info['data']

    H_full_list, S_full_list, matrix_size = _build_nonsoc_matrix_lists(
        H_R_dict, S_R_dict, lattice_vectors,
        params.num_ao or list(S_R_dict.values())[0].shape[0]
    )

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )

    # Create engine
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=os.path.join(BASE_DIR, 'mgb2_scdm_dummy'),
        num_wann=None,
        outer_window=(-5.0, 3.0),
        e_fermi=params.fermi_energy,
        window_is_relative=True,
    )

    # Parse atomic basis info
    atomic_info = parse_atomic_basis_info(lines)
    engine.atom_positions = atomic_info.atom_positions
    engine.basis_atom_map = atomic_info.basis_atom_map

    # Solve eigenvalue problems
    print("\nSolving eigenvalue problems...")
    engine.solve_all_kpoints(parallel=True)
    print(f"Done in {time.time() - t0:.1f}s")

    # Parse atoms for .win file
    atoms_result = parse_atoms_from_crystal_output(lines)
    atoms = atoms_result[0]

    # Use the same band selection as the scored configs:
    # bands 16-42 (27 bands), dis_win = [-14.47, 20.30] eV relative to E_F
    # These are the valence + low conduction bands, matching the scored configs.
    all_band_indices = np.arange(16, 43)  # bands 16..42 inclusive
    num_bands_in_window = len(all_band_indices)
    engine.selected_band_indices = all_band_indices
    print(f"Bands in window: {num_bands_in_window} (indices {all_band_indices[0]}-{all_band_indices[-1]})")

    # ================================================================
    # Generate SCDM-only configs
    # ================================================================
    for num_wann in SCDM_CONFIGS:
        seedname = os.path.join(BASE_DIR, f'mgb2_scdm{num_wann}')
        rel_seedname = f'mgb2_scdm{num_wann}'

        print(f"\n{'=' * 70}")
        print(f"SCDM-only: num_wann={num_wann} (seedname={rel_seedname})")
        print(f"{'=' * 70}")

        # Apply to engine
        engine.seedname = seedname
        engine.num_wann = num_wann
        engine.selected_band_indices = all_band_indices
        engine._num_bands_for_win = num_bands_in_window
        # Same disentanglement window as scored configs (absolute eV)
        engine._dis_win = (-14.46644607, 20.30264816)
        engine._dis_froz = None  # No frozen window (entangled metal)
        engine._override_num_iter = 5000

        # Pure SCDM: no orbital mask — all 55 AOs available for pivoting
        engine.select_projections(
            verbose=True, method='scdm', orbital_mask=None,
        )

        # Step A: Write .win file — no projections (uses 'random')
        t1 = time.time()
        win_config = create_win_config_from_engine(
            engine, atoms=atoms, projections=None, spinors=has_soc,
            write_hr=True, num_iter=5000,
        )
        write_win_file(seedname, win_config, verbose=True)

        # Step B: Run wannier90.x -pp to generate .nnkp
        print(f"\n  Running: wannier90.x -pp {rel_seedname}")
        pp_result = subprocess.run(
            [W90_BIN, '-pp', rel_seedname],
            capture_output=True, text=True, cwd=BASE_DIR,
        )
        if pp_result.returncode != 0:
            print(f"  -pp FAILED: {pp_result.stdout[:300]}")
            print(f"  stderr: {pp_result.stderr[:300]}")
            continue
        print(f"  .nnkp generated")

        # Step C: Write data files (.eig, .amn, .mmn) using .nnkp neighbors
        engine.write_files(
            verbose=True,
            write_win=False,
            use_nnkp=True,
            atoms=atoms,
            projections=None,
            spinors=has_soc,
        )
        print(f"  Files written in {time.time() - t1:.1f}s")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("\nTo run Wannier90:")
    print(f"  cd {BASE_DIR}")
    for nw in SCDM_CONFIGS:
        print(f"  wannier90.x mgb2_scdm{nw}")


if __name__ == '__main__':
    main()
