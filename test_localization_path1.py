#!/usr/bin/env python3
"""
Test Path 1: Automatic Projections with Disentanglement
========================================================

Configuration:
- num_wann = 12 (6 p-orbitals × 2 for spinors)
- num_bands = 16 (DFT bands 40-55)
- Projections: Bi:p (automatic)
- Disentanglement: ENABLED

This is the EASIEST approach - let Wannier90 do the work.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import (
    Wannier90Engine,
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    parse_atomic_basis_info,
    create_spin_block_matrices,
    parse_atoms_from_crystal_output,
)


def create_path1_test():
    """
    Path 1: Automatic projections with disentanglement
    """
    print("=" * 80)
    print("Path 1: Automatic Bi:p Projections with Disentanglement")
    print("=" * 80)

    input_file = "tests/Bismuth_basis_40.out"
    output_dir = "./test_output"
    seedname = "bismuth_path1_auto"

    # Parse input
    print("\n1. Parsing CRYSTAL output...")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    H_R_dict, S_R_dict, lattice_vectors_list, num_basis = parse_overlap_and_fock_matrices(lines)
    atoms, _ = parse_atoms_from_crystal_output(lines)
    atomic_info = parse_atomic_basis_info(lines)

    print(f"   ✓ Fermi energy: {params.fermi_energy:.3f} eV")
    print(f"   ✓ Atoms: {len(atoms)}, Basis: {num_basis}")

    # Create spin-block matrices
    print("\n2. Creating SOC matrices...")
    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict, S_R_dict, num_basis, lattice_vectors_list
    )
    num_orbitals = H_full_list[0][1].shape[0]
    print(f"   ✓ SOC matrix size: {num_orbitals}×{num_orbitals}")

    # Prepare real-space matrices
    print("\n3. Preparing real-space matrices...")
    lattice_vectors = lattice_vectors_list[0]
    real_space_matrices = {}

    for R_cart, H_mat in H_full_list:
        S_mat = None
        for S_R_cart, S_matrix in S_full_list:
            if np.allclose(R_cart, S_R_cart):
                S_mat = S_matrix
                break

        R_int = tuple(np.round(np.linalg.solve(lattice_vectors.T, R_cart)).astype(int))
        if S_mat is not None:
            real_space_matrices[R_int] = {'H': H_mat, 'S': S_mat}

    print(f"   ✓ {len(real_space_matrices)} R-vectors")

    # Initialize engine with outer window for disentanglement
    print("\n4. Initializing engine...")

    # Band selection: 40-55 (16 bands)
    # Outer window: -8.5 to 0.0 eV (covers bands 40-55)
    # Frozen window: -8.0 to -1.1 eV (original 10 bands)
    # Target: 12 WFs from 16 bands

    k_grid = params.k_grid if params.k_grid else (15, 15, 1)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=k_grid,
        lattice_vectors=lattice_vectors,
        num_wann=12,  # Target 12 WFs
        seedname=os.path.join(output_dir, seedname),
        outer_window=(-8.5, 0.0),  # Outer window for disentanglement
        e_fermi=params.fermi_energy,
        num_electrons=params.num_electrons
    )

    # Set atomic info for phase-corrected MMN
    if num_orbitals == 2 * atomic_info.num_basis:
        engine.atom_positions = atomic_info.atom_positions
        engine.basis_atom_map = np.concatenate([
            atomic_info.basis_atom_map,
            atomic_info.basis_atom_map
        ])
        print(f"   ✓ Atomic info set (SOC doubled)")

    # Solve eigenvalue problems
    print("\n5. Solving eigenvalue problems...")
    use_parallel = np.prod(k_grid) > 16
    engine.solve_all_kpoints(parallel=use_parallel)
    print(f"   ✓ Solved {engine.num_kpoints} k-points")

    # Analyze bands
    print("\n6. Analyzing bands...")
    engine.analyze_bands(verbose=True)

    # Write files
    print("\n7. Writing Wannier90 files...")

    # Convert atoms to correct format for win file
    atoms_frac = []
    for symbol, pos_cart in atoms:
        pos_frac = np.linalg.solve(lattice_vectors.T, pos_cart)
        atoms_frac.append((symbol, pos_frac))

    # Use automatic Bi:p projections
    projections = ["Bi:p"]

    engine.write_files(
        verbose=True,
        write_win=True,
        atoms=atoms_frac,
        projections=projections,
        spinors=True,
        bands_plot=False,
        kpoint_path=None
    )

    # Manually add disentanglement parameters to .win file
    print("\n8. Adding disentanglement parameters...")
    win_file = os.path.join(output_dir, f"{seedname}.win")

    with open(win_file, 'r') as f:
        lines = f.readlines()

    # Find where to insert disentanglement parameters (after num_bands line)
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('num_bands'):
            # Update num_bands
            lines[i] = f"num_bands = 16\n"
            insert_idx = i + 1
            break

    if insert_idx:
        disentanglement_lines = [
            "\n",
            "! Disentanglement windows\n",
            "dis_win_min = -8.5\n",
            "dis_win_max = 0.0\n",
            "dis_froz_min = -8.0\n",
            "dis_froz_max = -1.1\n",
            "dis_num_iter = 1000\n",
            "dis_mix_ratio = 0.5\n",
        ]
        lines[insert_idx:insert_idx] = disentanglement_lines

        with open(win_file, 'w') as f:
            f.writelines(lines)

        print(f"   ✓ Updated {win_file} with disentanglement")

    # Summary
    print("\n" + "=" * 80)
    print("✓ PATH 1 TEST SETUP COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Seedname: {seedname}")
    print(f"\nConfiguration:")
    print(f"  - num_wann = 12")
    print(f"  - num_bands = 16 (DFT bands 40-55)")
    print(f"  - Projections: Bi:p (automatic, 6 orbitals × 2 spinor = 12)")
    print(f"  - Disentanglement: ENABLED")
    print(f"  - Outer window: [-8.5, 0.0] eV")
    print(f"  - Frozen window: [-8.0, -1.1] eV")
    print(f"\nNext steps:")
    print(f"  1. Transfer files to cluster:")
    print(f"     scp {output_dir}/{seedname}.* cluster:~/test_path1/")
    print(f"  2. Run Wannier90:")
    print(f"     wannier90.x {seedname}")
    print(f"  3. Check convergence:")
    print(f"     grep 'CONV' {seedname}.wout | tail -20")
    print(f"     grep 'Final Spread' {seedname}.wout")
    print(f"\nExpected improvement:")
    print(f"  - Omega_OD: 559 → <100 Ang²")
    print(f"  - Individual spreads: 40-90 → <20 Ang²")
    print(f"  - Convergence: ~500-2000 iterations")
    print("=" * 80)

    return True


if __name__ == '__main__':
    try:
        create_path1_test()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
