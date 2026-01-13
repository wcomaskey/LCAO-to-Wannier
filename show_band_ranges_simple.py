#!/usr/bin/env python3
"""
Show Band Energy Ranges (Simple Version)

This script uses the Stage 2 output to show band energy ranges.
Run Stage 2 first with a WIDE window to get all bands.

Usage:
    # First run Stage 2 with very wide window
    python3 lcao_to_wannier90.py --stage 2 --input tests/Bismuth_basis_40.out --seedname temp --window -20 15

    # Then run this script
    python3 show_band_ranges_simple.py --seedname temp
"""

import sys
import argparse
from pathlib import Path

def parse_eig_file(eig_file):
    """Parse .eig file to get band energies."""
    bands_per_kpt = {}

    with open(eig_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                band_idx = int(parts[0])
                kpt_idx = int(parts[1])
                energy = float(parts[2])

                if band_idx not in bands_per_kpt:
                    bands_per_kpt[band_idx] = []
                bands_per_kpt[band_idx].append(energy)

    return bands_per_kpt


def show_band_ranges(seedname, num_wann=12, target_num_bands=18):
    """Display band ranges from .eig file."""
    eig_file = f"{seedname}.eig"

    if not Path(eig_file).exists():
        print(f"Error: {eig_file} not found!")
        print()
        print("Please run Stage 2 with a WIDE window first:")
        print(f"  python3 lcao_to_wannier90.py --stage 2 \\")
        print(f"    --input tests/Bismuth_basis_40.out \\")
        print(f"    --seedname {seedname} \\")
        print(f"    --window -20 15")
        sys.exit(1)

    print("=" * 80)
    print(f"BAND ENERGY RANGES FROM {eig_file}")
    print("=" * 80)
    print()

    bands = parse_eig_file(eig_file)
    num_bands = len(bands)
    num_kpts = len(bands[1])

    print(f"Number of bands: {num_bands}")
    print(f"Number of k-points: {num_kpts}")
    print()

    # Compute min/max for each band
    print("=" * 80)
    print("BAND ENERGY RANGES (relative to Fermi level)")
    print("=" * 80)
    print()
    print(f"{'Band':<6} {'Min (eV)':<12} {'Max (eV)':<12} {'Width':<10} {'Crosses E_F':<12} {'Notes'}")
    print("-" * 80)

    band_stats = []
    for band_idx in sorted(bands.keys()):
        energies = bands[band_idx]
        min_E = min(energies)
        max_E = max(energies)
        width = max_E - min_E
        crosses_ef = (min_E < 0) and (max_E > 0)

        band_stats.append((band_idx, min_E, max_E, crosses_ef))

        # Determine notes
        notes = ""
        if crosses_ef:
            notes = "← CROSSES FERMI"
        elif -6 < min_E < 6 and -6 < max_E < 6:
            notes = "← Near E_F"
        elif min_E < -15:
            notes = "← Deep valence"
        elif max_E > 10:
            notes = "← High conduction"

        cross_marker = "YES" if crosses_ef else ""

        print(f"{band_idx:<6} {min_E:>10.4f}   {max_E:>10.4f}   {width:>8.4f}   {cross_marker:<12} {notes}")

    print()

    # Find bands crossing Fermi
    fermi_bands = [b for b, min_E, max_E, crosses in band_stats if crosses]
    print(f"Bands crossing Fermi level: {len(fermi_bands)}")
    if fermi_bands:
        print(f"  Indices: {min(fermi_bands)} - {max(fermi_bands)}")
    print()

    # Recommendations
    print("=" * 80)
    print(f"WINDOW RECOMMENDATIONS FOR num_wann={num_wann}")
    print("=" * 80)
    print()

    # Find bands closest to Fermi
    import numpy as np
    band_distances = []
    for band_idx, min_E, max_E, _ in band_stats:
        dist = min(abs(min_E), abs(max_E))
        band_distances.append((dist, band_idx, min_E, max_E))

    band_distances.sort()  # Sort by distance from Fermi

    # Different recommendations
    targets = [
        (num_wann, "Minimal (no disentanglement)"),
        (int(num_wann * 1.5), "Recommended (150% of num_wann)"),
        (target_num_bands, f"Target ({target_num_bands} bands)"),
    ]

    for target, description in targets:
        if target > num_bands:
            continue

        selected = band_distances[:target]
        band_indices = [b[1] for b in selected]

        win_min = min(b[2] for b in selected) - 1.0  # 1 eV margin
        win_max = max(b[3] for b in selected) + 1.0

        froz_min = max(win_min, -6.0)
        froz_max = min(win_max, 4.0)

        print(f"{description}:")
        print(f"  num_bands = {target}")
        print(f"  dis_win_min = {win_min:.1f}")
        print(f"  dis_win_max = {win_max:.1f}")
        print(f"  dis_froz_min = {froz_min:.1f}")
        print(f"  dis_froz_max = {froz_max:.1f}")
        print(f"  Contains bands: {min(band_indices)} - {max(band_indices)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Show band energy ranges from .eig file'
    )
    parser.add_argument('--seedname', default='bismuth',
                       help='Seedname (default: bismuth)')
    parser.add_argument('--num-wann', type=int, default=12,
                       help='Number of Wannier functions (default: 12)')
    parser.add_argument('--target-bands', type=int, default=18,
                       help='Target number of DFT bands (default: 18)')

    args = parser.parse_args()

    show_band_ranges(args.seedname, args.num_wann, args.target_bands)


if __name__ == '__main__':
    main()
