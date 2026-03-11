#!/usr/bin/env python3
"""
Show Band Energy Ranges

This script displays the energy range of each band across all k-points,
helping you choose optimal energy windows for Wannier90.

Usage:
    python3 show_band_ranges.py --input tests/Bismuth_basis_40.out
"""

import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
)
from lcao_wannier.solver import solve_all_kpoints_parallel


def show_band_ranges(input_file, num_wann=12, target_num_bands=18):
    """
    Display energy ranges for all bands to help select windows.
    """
    print("=" * 80)
    print("BAND ENERGY RANGES ANALYSIS")
    print("=" * 80)
    print()

    # Parse CRYSTAL output
    print("Reading CRYSTAL output file...")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)

    num_kpoints = params.k_grid[0] * params.k_grid[1] * params.k_grid[2]

    print(f"System Information:")
    print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Total k-points: {num_kpoints}")
    print()

    # Solve eigenvalue problems
    print("Solving eigenvalue problems at all k-points...")
    eigenvalues_list, eigenvectors_list = solve_all_kpoints_parallel(raw_matrices)
    print(f"✓ Solved for {len(eigenvalues_list)} k-points")
    print()

    # Convert to eV relative to Fermi
    eigenvalues_eV_list = [
        (eigs - params.fermi_energy) * 27.211386  # Hartree to eV
        for eigs in eigenvalues_list
    ]

    # Get number of bands
    num_bands_total = eigenvalues_list[0].shape[0]
    print(f"Total number of bands: {num_bands_total}")
    print()

    # Compute min/max energy for each band across all k-points
    band_mins = np.zeros(num_bands_total)
    band_maxs = np.zeros(num_bands_total)

    for band_idx in range(num_bands_total):
        energies_this_band = [eigs[band_idx] for eigs in eigenvalues_eV_list]
        band_mins[band_idx] = min(energies_this_band)
        band_maxs[band_idx] = max(energies_this_band)

    # Display results
    print("=" * 80)
    print("BAND ENERGY RANGES (relative to Fermi level)")
    print("=" * 80)
    print()
    print(f"{'Band':<6} {'Min (eV)':<12} {'Max (eV)':<12} {'Width':<10} {'Crosses E_F':<12} {'Notes'}")
    print("-" * 80)

    for i in range(num_bands_total):
        min_E = band_mins[i]
        max_E = band_maxs[i]
        width = max_E - min_E
        crosses_ef = (min_E < 0) and (max_E > 0)

        # Determine notes
        notes = ""
        if crosses_ef:
            notes = "← CROSSES FERMI"
        elif abs(min_E) < 6 and abs(max_E) < 6:
            notes = "← Near E_F"
        elif min_E < -15:
            notes = "← Deep valence"
        elif max_E > 10:
            notes = "← High conduction"

        cross_marker = "YES" if crosses_ef else ""

        print(f"{i+1:<6} {min_E:>10.4f}   {max_E:>10.4f}   {width:>8.4f}   {cross_marker:<12} {notes}")

    print()
    print("=" * 80)
    print(f"WINDOW RECOMMENDATIONS FOR num_wann={num_wann}")
    print("=" * 80)
    print()

    # Find bands that cross Fermi level
    fermi_crossing_bands = [i for i in range(num_bands_total)
                           if band_mins[i] < 0 and band_maxs[i] > 0]

    print(f"Bands crossing Fermi level: {len(fermi_crossing_bands)}")
    if fermi_crossing_bands:
        print(f"  Indices: {fermi_crossing_bands[0]+1} - {fermi_crossing_bands[-1]+1}")
    print()

    # Suggest windows based on target_num_bands
    suggestions = [
        (num_wann, "Minimal (num_bands = num_wann, no disentanglement)"),
        (int(num_wann * 1.2), "Conservative (num_bands = 120% of num_wann)"),
        (int(num_wann * 1.5), "Recommended (num_bands = 150% of num_wann)"),
        (int(num_wann * 2.0), "Aggressive (num_bands = 200% of num_wann)"),
    ]

    print("Suggested window configurations:")
    print()

    for target, description in suggestions:
        # Find narrowest window that contains target bands near Fermi
        # Strategy: Find bands closest to Fermi level
        band_distances = np.minimum(np.abs(band_mins), np.abs(band_maxs))
        closest_bands = np.argsort(band_distances)[:target]

        # Get min/max of selected bands
        selected_mins = band_mins[closest_bands]
        selected_maxs = band_maxs[closest_bands]

        win_min = selected_mins.min()
        win_max = selected_maxs.max()

        # Add margin
        margin = 1.0  # eV
        win_min -= margin
        win_max += margin

        # Frozen window (narrower)
        froz_min = max(win_min, -6.0)
        froz_max = min(win_max, 4.0)

        print(f"{description}")
        print(f"  Target num_bands: {target}")
        print(f"  Outer window: [{win_min:.1f}, {win_max:.1f}] eV")
        print(f"  Frozen window: [{froz_min:.1f}, {froz_max:.1f}] eV")
        print(f"  Contains bands: {closest_bands[0]+1} - {closest_bands[-1]+1}")
        print()

    # Find window for exactly target_num_bands
    print("=" * 80)
    print(f"OPTIMAL WINDOW FOR num_bands = {target_num_bands}")
    print("=" * 80)
    print()

    band_distances = np.minimum(np.abs(band_mins), np.abs(band_maxs))
    closest_bands = np.argsort(band_distances)[:target_num_bands]
    closest_bands = np.sort(closest_bands)  # Sort for readability

    selected_mins = band_mins[closest_bands]
    selected_maxs = band_maxs[closest_bands]

    win_min = selected_mins.min() - 1.0  # 1 eV margin
    win_max = selected_maxs.max() + 1.0

    froz_min = max(win_min, -6.0)
    froz_max = min(win_max, 4.0)

    print(f"To capture {target_num_bands} bands closest to Fermi level:")
    print()
    print(f"Selected bands: {closest_bands[0]+1} to {closest_bands[-1]+1}")
    print(f"  Band {closest_bands[0]+1}: [{band_mins[closest_bands[0]]:.2f}, {band_maxs[closest_bands[0]]:.2f}] eV")
    print(f"  Band {closest_bands[-1]+1}: [{band_mins[closest_bands[-1]]:.2f}, {band_maxs[closest_bands[-1]]:.2f}] eV")
    print()
    print("Recommended .win configuration:")
    print(f"  num_wann = {num_wann}")
    print(f"  num_bands = {target_num_bands}")
    print(f"  dis_win_min = {win_min:.1f}")
    print(f"  dis_win_max = {win_max:.1f}")
    print(f"  dis_froz_min = {froz_min:.1f}")
    print(f"  dis_froz_max = {froz_max:.1f}")
    print()

    # Check how many bands are actually in the window
    bands_in_window = []
    for i in range(num_bands_total):
        if band_mins[i] <= win_max and band_maxs[i] >= win_min:
            bands_in_window.append(i)

    print(f"Verification:")
    print(f"  Window [{win_min:.1f}, {win_max:.1f}] eV contains {len(bands_in_window)} bands")
    print(f"  Indices: {bands_in_window[0]+1} - {bands_in_window[-1]+1}")
    print()

    if len(bands_in_window) < target_num_bands:
        print(f"⚠ WARNING: Window only contains {len(bands_in_window)} bands, but num_bands={target_num_bands}")
        print(f"  Stage 2 will use all {len(bands_in_window)} bands available")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Show band energy ranges to help choose Wannier90 windows'
    )
    parser.add_argument('--input', default='tests/Bismuth_basis_40.out',
                       help='CRYSTAL output file (default: tests/Bismuth_basis_40.out)')
    parser.add_argument('--num-wann', type=int, default=12,
                       help='Number of Wannier functions (default: 12)')
    parser.add_argument('--target-bands', type=int, default=18,
                       help='Target number of DFT bands (default: 18)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    show_band_ranges(args.input, args.num_wann, args.target_bands)


if __name__ == '__main__':
    main()
