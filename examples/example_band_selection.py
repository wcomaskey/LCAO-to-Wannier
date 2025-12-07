#!/usr/bin/env python3
"""
Example: LCAO to Wannier90 with Automatic Band Selection

This example demonstrates the new band selection features using
2D Bismuth with spin-orbit coupling.

Features demonstrated:
- Automatic Fermi level detection
- Energy window-based band selection
- Automatic projection orbital selection
- Comparison with full-basis calculation
"""

import sys
import argparse
import numpy as np

from lcao_wannier import (
    Wannier90Engine,
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    estimate_fermi_energy,
    analyze_band_window,
    print_band_analysis,
    suggest_optimal_window,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LCAO to Wannier90 with automatic band selection'
    )
    parser.add_argument(
        'lcao_file',
        help='Path to LCAO/CRYSTAL output file'
    )
    parser.add_argument(
        '--k-grid', '-k',
        type=int, nargs=3, default=[8, 8, 1],
        metavar=('NX', 'NY', 'NZ'),
        help='K-point grid dimensions (default: 8 8 1)'
    )
    parser.add_argument(
        '--window', '-w',
        type=float, nargs=2, default=None,
        metavar=('E_MIN', 'E_MAX'),
        help='Energy window relative to E_F in eV (e.g., -3.0 5.0)'
    )
    parser.add_argument(
        '--e-fermi', '-ef',
        type=float, default=None,
        help='Fermi energy in eV (auto-detected if not specified)'
    )
    parser.add_argument(
        '--num-electrons', '-ne',
        type=int, default=None,
        help='Number of electrons (helps with E_F detection)'
    )
    parser.add_argument(
        '--seedname', '-s',
        type=str, default='bismuth',
        help='Seedname for output files (default: bismuth)'
    )
    parser.add_argument(
        '--suggest-window',
        action='store_true',
        help='Analyze band structure and suggest optimal window'
    )
    parser.add_argument(
        '--target-wann',
        type=int, default=None,
        help='Target number of Wannier functions (for window suggestion)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification checks'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("LCAO TO WANNIER90 - BAND SELECTION EXAMPLE")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Parse LCAO output file
    # =========================================================================
    print(f"\nParsing LCAO file: {args.lcao_file}")
    print("=" * 70)
    
    # Read the file
    with open(args.lcao_file, 'r') as f:
        lines = f.readlines()
    print(f"  Read {len(lines)} lines from file")
    
    # Parse calculation parameters (Fermi energy, electron count, k-grid, etc.)
    params = parse_calculation_parameters(lines)
    
    if params.fermi_energy is not None:
        print(f"  Fermi energy from file: {params.fermi_energy:.4f} eV")
    if params.num_electrons is not None:
        print(f"  Number of electrons from file: {params.num_electrons}")
    if params.k_grid is not None:
        print(f"  K-grid from file: {params.k_grid[0]} × {params.k_grid[1]} × {params.k_grid[2]}")
    if params.num_ao is not None:
        print(f"  Number of AO from file: {params.num_ao}")
    
    # Use file values as defaults if not specified by user
    if args.e_fermi is None and params.fermi_energy is not None:
        args.e_fermi = params.fermi_energy
        print(f"  → Using Fermi energy from file: {args.e_fermi:.4f} eV")
    
    if args.num_electrons is None and params.num_electrons is not None:
        args.num_electrons = params.num_electrons
        print(f"  → Using electron count from file: {args.num_electrons}")
    
    # Check if user provided default k-grid [8, 8, 1] - if so, prefer file value
    default_k_grid = [8, 8, 1]
    if args.k_grid == default_k_grid and params.k_grid is not None:
        args.k_grid = list(params.k_grid)
        print(f"  → Using k-grid from file: {args.k_grid[0]} × {args.k_grid[1]} × {args.k_grid[2]}")
    
    # Parse matrices and lattice vectors
    matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    
    # Convert lattice vectors to numpy array
    lattice_vectors = np.array(lattice_vectors_list)
    
    print(f"  Found {len(matrices)} matrix blocks")
    print(f"  Lattice vectors shape: {lattice_vectors.shape}")
    print(f"  Lattice vectors (Angstrom):")
    print(f"    a1 = [{lattice_vectors[0,0]:8.4f}, {lattice_vectors[0,1]:8.4f}, {lattice_vectors[0,2]:8.4f}]")
    print(f"    a2 = [{lattice_vectors[1,0]:8.4f}, {lattice_vectors[1,1]:8.4f}, {lattice_vectors[1,2]:8.4f}]")
    print(f"    a3 = [{lattice_vectors[2,0]:8.4f}, {lattice_vectors[2,1]:8.4f}, {lattice_vectors[2,2]:8.4f}]")
    
    # =========================================================================
    # Step 2: Organize matrices by lattice vector and spin channel
    # =========================================================================
    print("\nOrganizing matrices...")
    
    # Build H_R_dict and S_R_dict from parsed matrices
    H_R_dict = {}  # {R: {spin_channel: complex_matrix}}
    S_R_dict = {}  # {R: matrix}
    N_basis = None
    
    for mat_info in matrices:
        R = tuple(mat_info['lattice_vector'])
        mat_type = mat_info['type']
        data = mat_info['data']
        
        if data is None:
            continue
            
        if mat_type == 'overlap':
            S_R_dict[R] = data
            if N_basis is None:
                N_basis = data.shape[0]
        elif mat_type == 'fock':
            spin_channel = mat_info['spin_channel']
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data
            if N_basis is None:
                N_basis = data.shape[0]
    
    print(f"  Basis size (orbitals per spin): {N_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")
    
    # Show spin channels
    spin_channels = set()
    for R, channels in H_R_dict.items():
        spin_channels.update(channels.keys())
    print(f"  Spin channels present: {sorted(spin_channels)}")
    
    # =========================================================================
    # Step 3: Create spin-block matrices
    # =========================================================================
    print("\nCreating spin-block matrices...")
    print("=" * 70)
    
    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict, S_R_dict, N_basis, lattice_vectors_list
    )
    
    num_orbitals = H_full_list[0][1].shape[0]  # (R_cart, H_matrix) tuple
    print(f"  Number of R-vectors: {len(H_full_list)}")
    print(f"  Matrix dimension: {num_orbitals} × {num_orbitals} (2N × 2N with SOC)")
    
    # =========================================================================
    # Step 4: Prepare real-space matrices for engine
    # =========================================================================
    # Convert from list of (R_cart, matrix) tuples to dict format
    # Note: We need to use integer R-vectors as keys, not Cartesian
    
    # Get the R-vectors from H_R_dict keys (integer format)
    r_vectors = list(H_R_dict.keys())
    
    # Build real_space_matrices dict matching R-vectors to their matrices
    # The H_full_list and S_full_list are ordered by the valid_pairs processing
    real_space_matrices = {}
    
    # Create a mapping from Cartesian R to integer R
    # We'll match based on the order they were processed
    h_idx = 0
    s_idx = 0
    
    for R_cart, H_mat in H_full_list:
        # Find the corresponding S matrix (same R_cart)
        S_mat = None
        for S_R_cart, S_matrix in S_full_list:
            if np.allclose(R_cart, S_R_cart):
                S_mat = S_matrix
                break
        
        # Convert Cartesian back to integer R-vector
        R_int = tuple(np.round(np.linalg.solve(lattice_vectors.T, R_cart)).astype(int))
        
        if S_mat is not None:
            real_space_matrices[R_int] = {'H': H_mat, 'S': S_mat}
    
    print(f"  Prepared {len(real_space_matrices)} real-space matrix pairs")
    
    # =========================================================================
    # Step 5: Determine workflow based on arguments
    # =========================================================================
    k_grid = tuple(args.k_grid)
    
    if args.suggest_window:
        # Mode 1: Just analyze and suggest window
        print("\n" + "=" * 70)
        print("BAND STRUCTURE ANALYSIS MODE")
        print("=" * 70)
        
        # Create engine without window to solve for all bands
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=k_grid,
            lattice_vectors=lattice_vectors,
            seedname=args.seedname
        )
        
        # Solve eigenvalue problems
        engine.solve_all_kpoints(parallel=True)
        
        # Estimate Fermi level
        e_fermi = estimate_fermi_energy(
            engine.eigenvalues_list,
            num_electrons=args.num_electrons,
            method='auto'
        )
        print(f"\nEstimated Fermi energy: {e_fermi:.4f} eV")
        
        # Use file E_F if available, otherwise use estimated
        if args.e_fermi is not None:
            e_fermi_display = args.e_fermi
            print(f"Using Fermi energy from file: {e_fermi_display:.4f} eV")
        else:
            e_fermi_display = e_fermi
        
        # Get energy statistics for physically relevant bands (near E_F)
        all_eigs = np.array(engine.eigenvalues_list)
        
        # Find bands within reasonable range of E_F (exclude vacuum states)
        # Consider bands within 50 eV of E_F as "physical"
        band_mins = np.min(all_eigs, axis=0)
        band_maxs = np.max(all_eigs, axis=0)
        physical_mask = (band_maxs > e_fermi_display - 50) & (band_mins < e_fermi_display + 50)
        physical_bands = np.where(physical_mask)[0]
        
        if len(physical_bands) > 0:
            phys_eigs = all_eigs[:, physical_bands]
            e_min_phys = np.min(phys_eigs)
            e_max_phys = np.max(phys_eigs)
            
            print(f"\nBand structure statistics (bands near E_F):")
            print(f"  Physical bands: {physical_bands[0]}-{physical_bands[-1]} ({len(physical_bands)} bands)")
            print(f"  Energy range: [{e_min_phys:.2f}, {e_max_phys:.2f}] eV")
            print(f"  Bandwidth: {e_max_phys - e_min_phys:.2f} eV")
            
            # Find gap near E_F
            occupied = band_maxs[band_maxs < e_fermi_display]
            unoccupied = band_mins[band_mins > e_fermi_display]
            if len(occupied) > 0 and len(unoccupied) > 0:
                vbm = np.max(occupied)
                cbm = np.min(unoccupied)
                gap = cbm - vbm
                if gap > 0 and gap < 20:  # Reasonable gap
                    print(f"  Band gap: {gap:.2f} eV (VBM={vbm:.2f}, CBM={cbm:.2f})")
                else:
                    print(f"  Band gap: metallic or semi-metallic")
            
            # Also show full range for reference
            print(f"\nFull eigenvalue range (including vacuum states):")
            print(f"  [{np.min(all_eigs):.2f}, {np.max(all_eigs):.2f}] eV")
        
        # Suggest window
        suggested = suggest_optimal_window(
            engine.eigenvalues_list,
            e_fermi=e_fermi_display,
            target_num_wann=args.target_wann
        )
        
        # Convert suggested (absolute) to relative for display
        suggested_relative = (suggested[0] - e_fermi_display, suggested[1] - e_fermi_display)
        
        # Test the suggested window
        print("\nTesting suggested window...")
        print(f"  Absolute: [{suggested[0]:.2f}, {suggested[1]:.2f}] eV")
        print(f"  Relative to E_F: [{suggested_relative[0]:.2f}, {suggested_relative[1]:.2f}] eV")
        
        analysis = analyze_band_window(
            engine.eigenvalues_list,
            outer_window=suggested,
            e_fermi=e_fermi_display,
            window_is_relative=False  # suggested is in absolute coordinates
        )
        print_band_analysis(analysis)
        
        print("\n" + "=" * 70)
        print("SUGGESTION COMPLETE")
        print("=" * 70)
        print(f"\nTo run with this window (relative to E_F), use:")
        print(f"  python {sys.argv[0]} {args.lcao_file} \\")
        print(f"    --k-grid {k_grid[0]} {k_grid[1]} {k_grid[2]} \\")
        print(f"    --window {suggested_relative[0]:.2f} {suggested_relative[1]:.2f}")
        
    elif args.window is not None:
        # Mode 2: Run with specified window
        print("\n" + "=" * 70)
        print("RUNNING WITH BAND SELECTION")
        print("=" * 70)
        print(f"Energy window: [{args.window[0]:.2f}, {args.window[1]:.2f}] eV relative to E_F")
        
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=k_grid,
            lattice_vectors=lattice_vectors,
            seedname=args.seedname,
            outer_window=tuple(args.window),
            e_fermi=args.e_fermi,
            num_electrons=args.num_electrons,
            window_is_relative=True
        )
        
        # Run full workflow with band selection
        results = engine.run(
            parallel=True,
            verify=not args.no_verify,
            analyze_window=True,
            select_orbitals=True
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        if results['band_analysis'] is not None:
            ba = results['band_analysis']
            print(f"Fermi energy: {ba.e_fermi:.4f} eV")
            print(f"Frozen bands: {ba.num_wann}")
            print(f"Frozen energy range: [{ba.frozen_energy_range[0]:.2f}, {ba.frozen_energy_range[1]:.2f}] eV")
            print(f"Band indices: {ba.frozen_indices[0]} to {ba.frozen_indices[-1]}")
        
        print(f"\nOutput files:")
        print(f"  {args.seedname}.eig - {results['num_wann']} bands × {engine.num_kpoints} k-points")
        print(f"  {args.seedname}.amn - Projection matrices")
        print(f"  {args.seedname}.mmn - Overlap matrices")
        
    else:
        # Mode 3: Run with full basis (original behavior)
        print("\n" + "=" * 70)
        print("RUNNING WITH FULL BASIS")
        print("=" * 70)
        print(f"Using all {num_orbitals} orbitals as Wannier functions")
        print("(Use --window to enable band selection)")
        
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=k_grid,
            lattice_vectors=lattice_vectors,
            seedname=args.seedname
        )
        
        results = engine.run(
            parallel=True,
            verify=not args.no_verify,
            analyze_window=False
        )
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Number of Wannier functions: {results['num_wann']}")
        print(f"\nOutput files:")
        print(f"  {args.seedname}.eig")
        print(f"  {args.seedname}.amn")
        print(f"  {args.seedname}.mmn")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()