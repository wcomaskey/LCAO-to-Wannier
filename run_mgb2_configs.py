#!/usr/bin/env python3
"""Run Wannier90 on top 7 MgB2 orbital configurations.

Shares the expensive eigenvalue solve and scoring across all configs.
For each config: writes .win/.eig/.amn/.mmn, runs wannier90.x.
"""

import sys, os, subprocess, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lcao_wannier.parser import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    parse_atomic_basis_info,
    parse_orbital_types,
)
from lcao_wannier.engine import Wannier90Engine
from lcao_wannier.utils import prepare_real_space_matrices
from lcao_wannier.symmetry import get_orbital_structure_from_crystal
from lcao_wannier.config_scoring import (
    evaluate_all_configurations,
    format_recommendation_table,
)
from lcao_wannier.win_file import parse_atoms_from_crystal_output
from lcao_to_wannier90 import (
    _build_nonsoc_matrix_lists,
    _get_per_atom_basis_counts,
    _infer_projections_from_config,
    _infer_projections,
)

INPUT_FILE = os.path.join(ROOT, 'MgB2/MgB2_basis_121.out')
BASE_DIR = os.path.join(ROOT, 'MgB2')
W90_BIN = os.path.join(ROOT, 'external/wannier90-3.1.0/wannier90.x')
N_PROCS = 6
TOP_N = 7  # Number of top configs to run


def main():
    t0 = time.time()

    # ================================================================
    # SHARED COMPUTATION (done once)
    # ================================================================
    print("=" * 70)
    print("SHARED SETUP: Parsing + Eigenvalue Solve")
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
        seedname=os.path.join(BASE_DIR, 'mgb2_dummy'),
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
    atom_symbols = [sym for sym, _ in atoms]

    # Parse orbital structure
    orbital_types_dict = parse_orbital_types(lines)
    per_atom_counts = _get_per_atom_basis_counts(atomic_info)
    orbital_structure = get_orbital_structure_from_crystal(
        orbital_types_dict, per_atom_counts, atomic_info.num_atoms,
    )
    # ================================================================
    # SCORING (done once)
    # ================================================================
    print("\n" + "=" * 70)
    print("SCORING: Evaluating all configurations")
    print("=" * 70)

    ranking = evaluate_all_configurations(
        engine.eigenvectors_list,
        engine.S_k_list,
        engine.eigenvalues_list,
        orbital_types_dict,
        orbital_structure,
        e_fermi=engine.e_fermi,
        has_soc=has_soc,
        verbose=True,
        kpoints=engine.kpoints,
        k_grid=engine.k_grid,
        lattice_vectors=engine.lattice_vectors,
        atom_positions=engine.atom_positions,
        basis_atom_map=engine.basis_atom_map,
        stacked=engine._stacked,
        atom_symbols=atom_symbols,
    )

    table = format_recommendation_table(ranking)
    print(table)
    print(f"\nScoring done in {time.time() - t0:.1f}s total")

    # Get top N configs
    configs = ranking.configurations[:TOP_N]
    print(f"\nWill run Wannier90 on top {TOP_N} configs:")
    for i, c in enumerate(configs):
        print(f"  {i+1}. {c.name} (num_wann={c.num_wann})")

    # ================================================================
    # PER-CONFIG: Write files + run Wannier90
    # ================================================================
    results = {}

    for idx, config in enumerate(configs):
        cfg_name = config.name.replace(':', '_').replace('+', '_')
        seedname = os.path.join(BASE_DIR, f'mgb2_{cfg_name}')
        rel_seedname = f'mgb2_{cfg_name}'  # Relative to BASE_DIR (Fortran 50-char limit)

        print(f"\n{'=' * 70}")
        print(f"CONFIG {idx+1}/{TOP_N}: {config.name} "
              f"(num_wann={config.num_wann}, seedname={cfg_name})")
        print(f"{'=' * 70}")

        # Apply config to engine
        engine.seedname = seedname
        engine.num_wann = config.num_wann

        if config.selected_band_indices is not None:
            engine.selected_band_indices = config.selected_band_indices
        engine._num_bands_for_win = config.num_bands_in_window
        engine._dis_win = config.recommended_dis_win
        engine._dis_froz = config.recommended_dis_froz
        engine._override_num_iter = 5000

        # SCDM projection selection with this config's orbital mask
        engine.select_projections(
            verbose=True, method='scdm', orbital_mask=config.orbital_mask,
        )

        # Determine projections for .win file
        if config.per_atom_types is not None:
            projections = _infer_projections_from_config(atoms, config.per_atom_types)
        else:
            projections = _infer_projections(atoms, config.num_wann, has_soc)

        print(f"  Projections: {projections}")

        # Step A: Write .win file only
        t1 = time.time()
        from lcao_wannier.win_file import create_win_config_from_engine, write_win_file
        win_config = create_win_config_from_engine(
            engine, atoms=atoms, projections=projections, spinors=has_soc,
            write_hr=True, num_iter=5000,
        )
        write_win_file(seedname, win_config, verbose=True)

        # Step B: Run wannier90.x -pp to generate .nnkp
        # Must use relative seedname: Fortran has 50-char path limit
        w90_bin_abs = os.path.abspath(W90_BIN)
        print(f"\n  Running: wannier90.x -pp {rel_seedname} (from {BASE_DIR})")
        pp_result = subprocess.run(
            [w90_bin_abs, '-pp', rel_seedname],
            capture_output=True, text=True, cwd=BASE_DIR,
        )
        if pp_result.returncode != 0:
            print(f"  -pp FAILED: {pp_result.stdout[:300]}")
            print(f"  stderr: {pp_result.stderr[:300]}")
            results[config.name] = {'status': 'FAILED (-pp)', 'time': 0}
            continue
        print(f"  .nnkp generated")

        # Step C: Write data files (.eig, .amn, .mmn) using .nnkp neighbors
        engine.write_files(
            verbose=True,
            write_win=False,  # Already written
            use_nnkp=True,    # Use Wannier90's neighbor list
            atoms=atoms,
            projections=projections,
            spinors=has_soc,
        )
        print(f"  Files written in {time.time() - t1:.1f}s")

        # Step D: Run Wannier90 (serial — binary not compiled with MPI)
        print(f"\n  Running: wannier90.x {rel_seedname}")
        t2 = time.time()
        result = subprocess.run(
            [w90_bin_abs, rel_seedname],
            capture_output=True, text=True, cwd=BASE_DIR,
        )
        w90_time = time.time() - t2

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")
            results[config.name] = {'status': 'FAILED', 'time': w90_time}
        else:
            print(f"  Wannier90 completed in {w90_time:.1f}s")
            # Parse spread from .wout file
            wout_file = f"{seedname}.wout"
            spread = parse_wout_spread(wout_file)
            results[config.name] = {
                'status': 'OK',
                'time': w90_time,
                'num_wann': config.num_wann,
                **spread,
            }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: Wannier90 Spread Comparison")
    print(f"{'=' * 70}")
    print(f"{'Config':<16} {'N_WF':>4} {'Omega_I':>10} {'Omega_D':>10} "
          f"{'Omega_OD':>10} {'Omega_T':>10} {'Omega/WF':>10} {'Status':<8}")
    print("-" * 80)

    for config in configs:
        r = results.get(config.name, {})
        if r.get('status') == 'OK':
            oi = r.get('omega_i', float('nan'))
            od = r.get('omega_d', float('nan'))
            ood = r.get('omega_od', float('nan'))
            ot = r.get('omega_total', float('nan'))
            per_wf = ot / config.num_wann if not np.isnan(ot) else float('nan')
            print(f"{config.name:<16} {config.num_wann:>4} {oi:>10.4f} {od:>10.4f} "
                  f"{ood:>10.4f} {ot:>10.4f} {per_wf:>10.4f} {'OK':<8}")
        else:
            print(f"{config.name:<16} {config.num_wann:>4} {'---':>10} {'---':>10} "
                  f"{'---':>10} {'---':>10} {'---':>10} {r.get('status','?'):<8}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


def parse_wout_spread(wout_file):
    """Extract final spread values from Wannier90 .wout file."""
    result = {
        'omega_i': float('nan'),
        'omega_d': float('nan'),
        'omega_od': float('nan'),
        'omega_total': float('nan'),
    }
    if not os.path.exists(wout_file):
        return result

    with open(wout_file, 'r') as f:
        for line in f:
            if 'Omega I      =' in line:
                try:
                    result['omega_i'] = float(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Omega D      =' in line:
                try:
                    result['omega_d'] = float(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Omega OD     =' in line:
                try:
                    result['omega_od'] = float(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Omega Total  =' in line:
                try:
                    result['omega_total'] = float(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass

    return result


if __name__ == '__main__':
    main()
