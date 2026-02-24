#!/usr/bin/env python3
"""
Complete Workflow Test for Bismuth_basis_40.out

This script demonstrates the full LCAO-to-Wannier90 workflow including:
1. Parsing CRYSTAL output
2. Automatic band selection
3. Fermi level detection
4. Energy window analysis
5. Complete Wannier90 file generation (.win, .eig, .amn, .mmn)
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import (
    Wannier90Engine,
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    parse_atomic_basis_info,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    estimate_fermi_energy,
    suggest_optimal_window,
    analyze_band_window,
    parse_atoms_from_crystal_output,
    KPATH_HEXAGONAL_2D,
)


def test_bismuth_workflow(
    mode='full',
    k_grid=None,
    energy_window=None,
    target_wann=None,
    seedname='bismuth_test',
    output_dir='./test_output'
):
    """
    Run complete workflow on Bismuth_basis_40.out file.

    Parameters
    ----------
    mode : str
        'full' - Use all bands (no selection)
        'window' - Use energy window for band selection
        'suggest' - Analyze and suggest optimal window
    k_grid : tuple, optional
        K-point grid (default: use from file or (4, 4, 1) for testing)
    energy_window : tuple, optional
        (E_min, E_max) in eV relative to E_F
    target_wann : int, optional
        Target number of Wannier functions for suggestion mode
    seedname : str
        Output file prefix
    output_dir : str
        Output directory for generated files
    """

    # File path
    bismuth_file = os.path.join(
        os.path.dirname(__file__),
        'Bismuth_basis_40.out'
    )

    if not os.path.exists(bismuth_file):
        print(f"ERROR: File not found: {bismuth_file}")
        return False

    print("=" * 80)
    print("LCAO-to-Wannier90 Test: Bismuth with SOC (basis=40)")
    print("=" * 80)
    print(f"Input file: {bismuth_file}")
    print(f"Mode: {mode}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Parse CRYSTAL output
    print("Step 1: Parsing CRYSTAL output file...")
    print("-" * 80)

    with open(bismuth_file, 'r') as f:
        lines = f.readlines()

    # Parse calculation parameters
    params = parse_calculation_parameters(lines)
    print(f"✓ Parsed calculation parameters:")
    if params:
        if params.fermi_energy is not None:
            print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
        if params.num_electrons is not None:
            print(f"  Electrons: {params.num_electrons}")
        if params.k_grid is not None:
            print(f"  K-grid from file: {params.k_grid}")
        if params.num_ao is not None:
            print(f"  Number of AOs: {params.num_ao}")
        if params.num_atoms is not None:
            print(f"  Number of atoms: {params.num_atoms}")
        if params.total_energy is not None:
            print(f"  Total energy: {params.total_energy:.6f} Hartree")

    # Parse matrices
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)

    # Convert to numpy array
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed matrices:")
    print(f"  Lattice vectors shape: {lattice_vectors.shape}")
    print(f"  Number of matrix blocks: {len(raw_matrices)}")

    # Organize matrices by R-vector and spin channel (following example_band_selection.py)
    H_R_dict = {}  # {R: {spin_channel: matrix}}
    S_R_dict = {}  # {R: matrix}
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
            spin_channel = mat_info['spin_channel']
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data
            if num_basis is None:
                num_basis = data.shape[0]

    print(f"  Basis size (per spin): {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Parse atoms
    atoms, _ = parse_atoms_from_crystal_output(lines)
    if atoms:
        print(f"✓ Parsed {len(atoms)} atoms:")
        for symbol, pos in atoms[:3]:
            print(f"  {symbol}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        if len(atoms) > 3:
            print(f"  ... and {len(atoms)-3} more")

    # Parse atomic basis information for phase-corrected MMN
    try:
        atomic_info = parse_atomic_basis_info(lines)
        print(f"✓ Parsed atomic basis info: {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction")
        atomic_info = None
    print()

    # Step 2: Create spin-block matrices
    print("Step 2: Creating spin-block matrices for SOC...")
    print("-" * 80)

    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict,
        S_R_dict,
        num_basis,
        lattice_vectors_list
    )

    num_orbitals = H_full_list[0][1].shape[0]  # (R_cart, H_matrix) tuple
    print(f"✓ Created spin-block matrices: {num_orbitals}×{num_orbitals} (2N×2N with SOC)")
    print(f"  Number of R-vectors: {len(H_full_list)}")
    print()

    # Step 3: Prepare real-space matrices
    print("Step 3: Preparing real-space matrices...")
    print("-" * 80)

    # Convert from list of (R_cart, matrix) tuples to dict format
    real_space_matrices = {}
    for R_cart, H_mat in H_full_list:
        # Find the corresponding S matrix
        S_mat = None
        for S_R_cart, S_matrix in S_full_list:
            if np.allclose(R_cart, S_R_cart):
                S_mat = S_matrix
                break

        # Convert Cartesian back to integer R-vector
        R_int = tuple(np.round(np.linalg.solve(lattice_vectors.T, R_cart)).astype(int))

        if S_mat is not None:
            real_space_matrices[R_int] = {'H': H_mat, 'S': S_mat}

    print(f"✓ Organized {len(real_space_matrices)} R-vectors")
    print(f"  Total matrix dimension: {next(iter(real_space_matrices.values()))['H'].shape[0]}")
    print()

    # Step 4: Initialize engine
    print("Step 4: Initializing Wannier90Engine...")
    print("-" * 80)

    # Use k-grid from file or parameter or default
    if k_grid is None:
        if params and params.k_grid:
            k_grid = params.k_grid
            print(f"Using k-grid from CRYSTAL file: {k_grid}")
        else:
            k_grid = (4, 4, 1)  # Small grid for testing
            print(f"Using default test k-grid: {k_grid}")
            print("  (Use k_grid parameter for different grid)")

    # Determine num_wann based on mode
    if mode == 'full':
        num_wann = H_full_list[0][1].shape[0]  # (R_cart, H_matrix) tuple
        outer_window = None
        num_electrons = None
    elif mode == 'window':
        if energy_window is None:
            energy_window = (-5.0, 3.0)
            print(f"Using default energy window: {energy_window} eV")
        outer_window = energy_window
        num_wann = None  # Will be determined from window
        num_electrons = params.num_electrons if params else None
    elif mode == 'suggest':
        num_wann = None
        outer_window = None
        num_electrons = params.num_electrons if params else None

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=k_grid,
        lattice_vectors=lattice_vectors,
        num_wann=num_wann,
        seedname=os.path.join(output_dir, seedname),
        outer_window=outer_window,
        e_fermi=params.fermi_energy if params else None,
        num_electrons=num_electrons
    )

    # Set atomic information for phase-corrected MMN
    if atomic_info is not None:
        engine.atom_positions = atomic_info.atom_positions
        # For SOC systems, double the basis_atom_map (spin up + spin down)
        if num_orbitals == 2 * atomic_info.num_basis:
            # SOC system: duplicate the mapping for spin-up and spin-down
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,  # Spin up
                atomic_info.basis_atom_map   # Spin down
            ])
            print(f"✓ Atomic info added to engine for phase-corrected MMN (SOC system, doubled basis map)")
        else:
            # Non-SOC system
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Atomic info added to engine for phase-corrected MMN")

    print()

    # Step 5: Solve eigenvalue problems
    print("Step 5: Solving eigenvalue problems...")
    print("-" * 80)

    # Use parallel for larger grids
    use_parallel = np.prod(k_grid) > 16

    engine.solve_all_kpoints(parallel=use_parallel)
    print()

    # Step 6: Mode-specific analysis
    if mode == 'suggest':
        print("Step 6: Suggesting optimal energy window...")
        print("-" * 80)

        # Estimate Fermi energy if not provided
        if engine.e_fermi is None:
            e_fermi = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=num_electrons,
                method='auto'
            )
            print(f"✓ Auto-detected Fermi energy: {e_fermi:.4f} eV")
        else:
            e_fermi = engine.e_fermi
            print(f"✓ Using Fermi energy from file: {e_fermi:.4f} eV")

        # Suggest window
        if target_wann is None:
            target_wann = 20
            print(f"Using default target num_wann: {target_wann}")

        e_min, e_max = suggest_optimal_window(
            engine.eigenvalues_list,
            e_fermi=e_fermi,
            target_num_wann=target_wann,
            padding=0.5
        )

        # Convert to relative energies
        e_min_rel = e_min - e_fermi
        e_max_rel = e_max - e_fermi

        print(f"\n✓ Window suggestion for ~{target_wann} Wannier functions:")
        print(f"  Suggested outer window: [{e_min:.2f}, {e_max:.2f}] eV (absolute)")
        print(f"  Relative to E_F: [{e_min_rel:.2f}, {e_max_rel:.2f}] eV")

        # Analyze what this window captures
        test_result = analyze_band_window(
            engine.eigenvalues_list,
            (e_min, e_max),
            e_fermi,
            window_is_relative=False  # Window is in absolute energies
        )
        print(f"  Number of frozen bands: {test_result.num_wann}")

        print("\nTo use this window, run:")
        print(f"  ./run_tests.sh --mode window --window {e_min_rel:.1f} {e_max_rel:.1f}")
        print()

        return True  # Don't write files in suggest mode

    elif mode == 'window':
        print("Step 6: Analyzing energy window and selecting bands...")
        print("-" * 80)

        engine.analyze_bands(verbose=True)
        print()

        if engine.band_analysis.num_wann == 0:
            print("WARNING: No bands found in window!")
            print("Try expanding the energy window.")
            return False

    else:  # mode == 'full'
        print("Step 6: Using full basis (no band selection)...")
        print("-" * 80)
        print(f"✓ Using all {engine.num_wann} bands")
        print()

    # Step 7: Write Wannier90 files
    print("Step 7: Writing Wannier90 input files...")
    print("-" * 80)

    engine.write_files(
        verbose=True,
        write_win=True,
        atoms=atoms,
        projections=None,  # Use random (from .amn file)
        spinors=True,      # Bismuth with SOC
        bands_plot=False,
        kpoint_path=None   # Can add KPATH_HEXAGONAL_2D for 2D Bi
    )
    print()

    # Step 8: Summary
    print("=" * 80)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Output files in: {output_dir}/")
    print(f"  {seedname}.win  - Wannier90 input parameters")
    print(f"  {seedname}.eig  - Band energies")
    print(f"  {seedname}.amn  - Projection matrices")
    print(f"  {seedname}.mmn  - Overlap matrices")
    print()

    if mode == 'window' and engine.band_analysis:
        print("Band selection summary:")
        print(f"  Selected bands: {engine.band_analysis.num_wann}")
        print(f"  Energy range: [{engine.band_analysis.frozen_energy_range[0]:.2f}, "
              f"{engine.band_analysis.frozen_energy_range[1]:.2f}] eV")
        print()

    print("Next steps:")
    print(f"  cd {output_dir}")
    print(f"  wannier90.x {seedname}")
    print()

    return True


def main():
    """Main entry point with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test LCAO-to-Wannier90 workflow on Bismuth_basis_40.out',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full basis (all bands)
  python test_bismuth_full_workflow.py --mode full

  # With energy window
  python test_bismuth_full_workflow.py --mode window --window -5.0 3.0

  # Suggest optimal window
  python test_bismuth_full_workflow.py --mode suggest --target-wann 20

  # Custom k-grid
  python test_bismuth_full_workflow.py --mode window --k-grid 8 8 1 --window -3 2
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'window', 'suggest'],
        default='window',
        help='Workflow mode (default: window)'
    )

    parser.add_argument(
        '--k-grid',
        type=int,
        nargs=3,
        metavar=('NX', 'NY', 'NZ'),
        help='K-point grid (default: from file or 4 4 1)'
    )

    parser.add_argument(
        '--window',
        type=float,
        nargs=2,
        metavar=('E_MIN', 'E_MAX'),
        help='Energy window in eV relative to E_F (for window mode)'
    )

    parser.add_argument(
        '--target-wann',
        type=int,
        metavar='N',
        help='Target number of Wannier functions (for suggest mode)'
    )

    parser.add_argument(
        '--seedname',
        default='bismuth_test',
        help='Output file prefix (default: bismuth_test)'
    )

    parser.add_argument(
        '--output-dir',
        default='./test_output',
        help='Output directory (default: ./test_output)'
    )

    args = parser.parse_args()

    # Run workflow
    success = test_bismuth_workflow(
        mode=args.mode,
        k_grid=tuple(args.k_grid) if args.k_grid else None,
        energy_window=tuple(args.window) if args.window else None,
        target_wann=args.target_wann,
        seedname=args.seedname,
        output_dir=args.output_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
