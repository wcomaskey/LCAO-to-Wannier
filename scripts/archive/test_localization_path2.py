#!/usr/bin/env python3
"""
Test Path 2: Manual Projections (10 WFs, no disentanglement)
=============================================================

Configuration:
- num_wann = 10
- num_bands = 10 (DFT bands 40-49, frozen window)
- Projections: Manually specified 5 orbitals × 2 spinor = 10 WFs
- Disentanglement: DISABLED

This approach requires knowing which orbitals contribute to bands 40-49.
For Bismuth 2D, likely p-orbitals.
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


def create_path2_test():
    """
    Path 2: Manual projections without disentanglement
    """
    print("=" * 80)
    print("Path 2: Manual Projections (10 WFs, No Disentanglement)")
    print("=" * 80)

    input_file = "tests/Bismuth_basis_40.out"
    output_dir = "./test_output"
    seedname = "bismuth_path2_manual"

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

    # Get atomic positions in fractional coordinates
    atoms_frac = []
    for symbol, pos_cart in atoms:
        pos_frac = np.linalg.solve(lattice_vectors.T, pos_cart)
        atoms_frac.append((symbol, pos_frac))

    print(f"\n   Atomic positions (fractional):")
    for i, (symbol, pos) in enumerate(atoms_frac):
        print(f"     Atom {i+1} ({symbol}): [{pos[0]:8.6f}, {pos[1]:8.6f}, {pos[2]:8.6f}]")

    # Initialize engine with energy window (no disentanglement)
    print("\n4. Initializing engine...")

    k_grid = params.k_grid if params.k_grid else (15, 15, 1)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=k_grid,
        lattice_vectors=lattice_vectors,
        num_wann=None,  # Will be set by band analysis
        seedname=os.path.join(output_dir, seedname),
        outer_window=(-5.0, 3.0),  # Same as original
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

    # Manual projections: 5 orbitals × 2 spinor = 10 WFs
    # We'll provide 3 options and user can choose

    # Option A: 3 p-orbitals on atom 1, 2 on atom 2
    projections_A = [
        f"f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:px",
        f"f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:py",
        f"f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:pz",
        f"f={atoms_frac[1][1][0]:.6f},{atoms_frac[1][1][1]:.6f},{atoms_frac[1][1][2]:.6f}:px",
        f"f={atoms_frac[1][1][0]:.6f},{atoms_frac[1][1][1]:.6f},{atoms_frac[1][1][2]:.6f}:py",
    ]

    # Use Option A by default
    projections = projections_A

    engine.write_files(
        verbose=True,
        write_win=True,
        atoms=atoms_frac,
        projections=projections,
        spinors=True,
        bands_plot=False,
        kpoint_path=None
    )

    # Add alternative projection options to .win file as comments
    print("\n8. Adding projection alternatives to .win file...")
    win_file = os.path.join(output_dir, f"{seedname}.win")

    with open(win_file, 'r') as f:
        lines = f.readlines()

    # Find projections section and add alternatives
    proj_start = None
    proj_end = None
    for i, line in enumerate(lines):
        if 'begin projections' in line.lower():
            proj_start = i
        if 'end projections' in line.lower():
            proj_end = i
            break

    if proj_start and proj_end:
        # Create alternative projection options
        alternatives = [
            "\n",
            "! Alternative Projection Options (comment/uncomment to switch)\n",
            "! ============================================================\n",
            "!\n",
            "! OPTION B: Symmetric p-orbitals (px, py on both atoms, pz only on atom 1)\n",
            f"! f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:px\n",
            f"! f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:py\n",
            f"! f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:pz\n",
            f"! f={atoms_frac[1][1][0]:.6f},{atoms_frac[1][1][1]:.6f},{atoms_frac[1][1][2]:.6f}:px\n",
            f"! f={atoms_frac[1][1][0]:.6f},{atoms_frac[1][1][1]:.6f},{atoms_frac[1][1][2]:.6f}:py\n",
            "!\n",
            "! OPTION C: Mixed s-p character (if bands have s-character)\n",
            f"! f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:s\n",
            f"! f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:px\n",
            f"! f={atoms_frac[0][1][0]:.6f},{atoms_frac[0][1][1]:.6f},{atoms_frac[0][1][2]:.6f}:py\n",
            f"! f={atoms_frac[1][1][0]:.6f},{atoms_frac[1][1][1]:.6f},{atoms_frac[1][1][2]:.6f}:px\n",
            f"! f={atoms_frac[1][1][0]:.6f},{atoms_frac[1][1][1]:.6f},{atoms_frac[1][1][2]:.6f}:py\n",
            "!\n",
            "! NOTE: Each projection creates 2 WFs (spin up + spin down)\n",
            "! Total: 5 projections × 2 (spinor) = 10 WFs\n",
        ]

        lines[proj_end+1:proj_end+1] = alternatives

        with open(win_file, 'w') as f:
            f.writelines(lines)

        print(f"   ✓ Added projection alternatives to {win_file}")

    # Summary
    print("\n" + "=" * 80)
    print("✓ PATH 2 TEST SETUP COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Seedname: {seedname}")
    print(f"\nConfiguration:")
    print(f"  - num_wann = 10")
    print(f"  - num_bands = 10 (DFT bands 40-49, frozen)")
    print(f"  - Projections: Manual (5 orbitals × 2 spinor = 10)")
    print(f"  - Disentanglement: DISABLED")
    print(f"  - Default projections: 3 p-orbitals on atom 1, 2 on atom 2")
    print(f"\nProjections used (Option A):")
    for i, proj in enumerate(projections, 1):
        print(f"  {i}. {proj}")
    print(f"\nAlternative options are commented in the .win file")
    print(f"\nNext steps:")
    print(f"  1. (Optional) Edit {seedname}.win to try different projections")
    print(f"  2. Transfer files to cluster:")
    print(f"     scp {output_dir}/{seedname}.* cluster:~/test_path2/")
    print(f"  3. Run Wannier90:")
    print(f"     wannier90.x {seedname}")
    print(f"  4. Check convergence:")
    print(f"     grep 'CONV' {seedname}.wout | tail -20")
    print(f"     grep 'Final Spread' {seedname}.wout")
    print(f"\nExpected improvement:")
    print(f"  - Omega_OD: 559 → <150 Ang² (may need projection tuning)")
    print(f"  - Individual spreads: 40-90 → <25 Ang²")
    print(f"  - If not converging well, try PATH 1 with disentanglement")
    print("=" * 80)

    return True


if __name__ == '__main__':
    try:
        create_path2_test()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
