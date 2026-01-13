#!/usr/bin/env python3
"""
Find Energy Window for Target Number of Bands

This script determines what energy window is needed to capture
a specific number of DFT bands (e.g., 18 bands for num_wann=12).
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def find_window_for_target_bands(input_file, target_num_bands=18):
    """
    Binary search to find energy window that contains target_num_bands.
    """
    print("=" * 80)
    print(f"FINDING ENERGY WINDOW FOR {target_num_bands} BANDS")
    print("=" * 80)
    print()

    # Import here to avoid issues
    from lcao_wannier import (
        parse_calculation_parameters,
        parse_overlap_and_fock_matrices,
    )
    from lcao_wannier.solver import solve_all_kpoints_parallel

    # Parse CRYSTAL output
    with open(input_file, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)

    print(f"System Information:")
    print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    print(f"  K-grid: {params.k_grid}")
    print()

    # Solve eigenvalue problems
    print("Solving eigenvalue problems to get band structure...")
    eigenvalues_list, eigenvectors_list = solve_all_kpoints_parallel(raw_matrices)
    print(f"✓ Solved for {len(eigenvalues_list)} k-points")
    print()

    # Convert to eV relative to Fermi
    eigenvalues_eV_list = [
        (eigs - params.fermi_energy) * 27.211386  # Hartree to eV
        for eigs in eigenvalues_list
    ]

    # Test progressively wider windows
    print("=" * 80)
    print("TESTING ENERGY WINDOWS")
    print("=" * 80)
    print()

    test_windows = [
        (-5.0, 3.0),
        (-7.0, 5.0),
        (-10.0, 6.0),
        (-12.0, 8.0),
        (-15.0, 10.0),
        (-20.0, 15.0),
    ]

    best_window = None
    best_diff = float('inf')

    for win_min, win_max in test_windows:
        # Count bands in window across all k-points
        bands_counts = []

        for eigs_eV in eigenvalues_eV_list:
            in_window = np.sum((eigs_eV >= win_min) & (eigs_eV <= win_max))
            bands_counts.append(in_window)

        min_bands = min(bands_counts)
        max_bands = max(bands_counts)
        avg_bands = np.mean(bands_counts)

        print(f"Window [{win_min:>6.1f}, {win_max:>6.1f}] eV:")
        print(f"  Bands: {min_bands:3d} - {max_bands:3d} (avg: {avg_bands:5.1f})")

        # Check if this window works
        if min_bands >= target_num_bands:
            diff = min_bands - target_num_bands
            status = f"✓ WORKS (contains {min_bands} ≥ {target_num_bands})"

            if diff < best_diff:
                best_diff = diff
                best_window = (win_min, win_max)
                print(f"  {status} ← BEST SO FAR")
            else:
                print(f"  {status}")
        else:
            deficit = target_num_bands - min_bands
            print(f"  ✗ Too narrow (need {deficit} more bands)")

        print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if best_window:
        win_min, win_max = best_window
        print(f"✓ Use window: [{win_min:.1f}, {win_max:.1f}] eV")
        print()

        # Show what this gives at Gamma point
        gamma_eigs = eigenvalues_eV_list[0]
        in_window = np.where((gamma_eigs >= win_min) & (gamma_eigs <= win_max))[0]

        print(f"At Gamma point, this window contains {len(in_window)} bands:")
        print(f"  Indices: {in_window[0]+1} - {in_window[-1]+1} (1-indexed)")
        print(f"  Energies: [{gamma_eigs[in_window[0]]:.2f}, {gamma_eigs[in_window[-1]]:.2f}] eV")
        print()

        # Check distribution around Fermi
        below_ef = np.sum(gamma_eigs[in_window] < -0.1)
        crossing = np.sum((gamma_eigs[in_window] >= -0.1) & (gamma_eigs[in_window] <= 0.1))
        above_ef = np.sum(gamma_eigs[in_window] > 0.1)

        print(f"  Band distribution at Gamma:")
        print(f"    {below_ef} below E_F")
        print(f"    {crossing} crossing E_F")
        print(f"    {above_ef} above E_F")
        print()

        print("Configuration for bismuth.win:")
        print(f"  dis_win_min = {win_min:.1f}")
        print(f"  dis_win_max = {win_max:.1f}")
        print(f"  dis_froz_min = {max(win_min, -6.0):.1f}")
        print(f"  dis_froz_max = {min(win_max, 4.0):.1f}")
        print()
    else:
        print(f"✗ No window found that contains {target_num_bands} bands!")
        print(f"  Try a smaller target_num_bands value")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Find energy window for target number of bands')
    parser.add_argument('--target-bands', type=int, default=18,
                       help='Target number of DFT bands (default: 18)')
    args = parser.parse_args()

    find_window_for_target_bands('tests/Bismuth_basis_40.out', args.target_bands)
