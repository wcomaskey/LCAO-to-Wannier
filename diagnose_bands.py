#!/usr/bin/env python3
"""
Diagnose Band Selection - Find Optimal Energy Window

This script analyzes the band structure to find which energy window
gives physically meaningful bands for Wannier90.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    parse_atoms_from_crystal_output,
)
from lcao_wannier.engine import Wannier90Engine

def analyze_bands(input_file, num_wann=12):
    """
    Analyze band structure to find optimal configuration.
    """
    print("=" * 80)
    print("BAND STRUCTURE ANALYSIS")
    print("=" * 80)
    print()

    # Parse CRYSTAL output
    with open(input_file, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)

    print(f"System Information:")
    print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    print(f"  K-grid: {params.k_grid}")
    print()

    # Initialize engine to solve eigenvalue problems
    atoms, _ = parse_atoms_from_crystal_output(lines)

    # Create a temporary engine instance
    engine = Wannier90Engine(
        input_file=input_file,
        seedname="temp",
        num_wann=num_wann,
        fermi_energy=params.fermi_energy
    )

    # Solve eigenvalue problems
    print("Solving eigenvalue problems...")
    engine.solve_eigenvalue_problems(verbose=False)
    print(f"✓ Solved for {engine.num_kpoints} k-points")
    print()

    # Get eigenvalues at Gamma point (k=0,0,0) for analysis
    gamma_idx = 0
    eigenvalues = engine.eigenvalues_list[gamma_idx]

    # Convert to eV relative to Fermi
    eigenvalues_eV = eigenvalues - params.fermi_energy

    print("=" * 80)
    print("BAND ENERGIES AT GAMMA POINT (relative to E_F)")
    print("=" * 80)
    print(f"{'Band':<6} {'Energy (eV)':<15} {'Distance from E_F (eV)':<25} {'Notes'}")
    print("-" * 80)

    for i, E in enumerate(eigenvalues_eV, 1):
        dist = abs(E)
        notes = ""

        if abs(E) < 0.5:
            notes = "← CROSSES E_F"
        elif -6 < E < 3:
            notes = "← Near E_F (relevant)"
        elif E < -15:
            notes = "← Deep valence (core?)"

        print(f"{i:<6} {E:>13.6f}   {dist:>13.6f}            {notes}")

    print()
    print("=" * 80)
    print("TESTING DIFFERENT ENERGY WINDOWS")
    print("=" * 80)
    print()

    # Test different windows
    test_windows = [
        ("Narrow", -5.0, 3.0),
        ("Medium", -7.0, 5.0),
        ("Wide", -10.0, 6.0),
        ("Very Wide", -12.0, 8.0),
        ("Current", -5.7, 5.7),
    ]

    for name, win_min, win_max in test_windows:
        # Count bands in window across all k-points
        bands_in_window = []

        for k_idx in range(engine.num_kpoints):
            eigs = engine.eigenvalues_list[k_idx] - params.fermi_energy
            in_window = np.where((eigs >= win_min) & (eigs <= win_max))[0]
            bands_in_window.append(len(in_window))

        min_bands = min(bands_in_window)
        max_bands = max(bands_in_window)
        avg_bands = np.mean(bands_in_window)

        # Get band indices at Gamma
        in_window_gamma = np.where((eigenvalues_eV >= win_min) & (eigenvalues_eV <= win_max))[0]

        # Check if bands span Fermi level
        if len(in_window_gamma) > 0:
            energies_in_window = eigenvalues_eV[in_window_gamma]
            spans_fermi = (energies_in_window.min() < 0) and (energies_in_window.max() > 0)
        else:
            spans_fermi = False

        print(f"{name} Window: [{win_min:.1f}, {win_max:.1f}] eV")
        print(f"  Bands: {min_bands}-{max_bands} (avg: {avg_bands:.1f})")
        print(f"  At Gamma: {len(in_window_gamma)} bands (indices {in_window_gamma[0]+1}-{in_window_gamma[-1]+1})" if len(in_window_gamma) > 0 else "  At Gamma: 0 bands")
        print(f"  Spans Fermi level: {'✓ YES' if spans_fermi else '✗ NO'}")

        if len(in_window_gamma) >= num_wann:
            # Select bands closest to Fermi
            distances = np.abs(eigenvalues_eV[in_window_gamma])
            closest_indices = np.argsort(distances)[:num_wann]
            selected_bands = in_window_gamma[closest_indices]
            selected_energies = eigenvalues_eV[selected_bands]

            print(f"  Selected {num_wann} bands: {selected_bands[0]+1}-{selected_bands[-1]+1} (1-indexed)")
            print(f"  Energy range: [{selected_energies.min():.2f}, {selected_energies.max():.2f}] eV")

            # Count how many below/above Fermi
            below_ef = np.sum(selected_energies < -0.1)
            crossing = np.sum((selected_energies >= -0.1) & (selected_energies <= 0.1))
            above_ef = np.sum(selected_energies > 0.1)
            print(f"  Distribution: {below_ef} below E_F, {crossing} crossing, {above_ef} above E_F")

        print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("For Bi surface states (num_wann=12), recommended window:")
    print("  [-10.0, 6.0] eV")
    print("  This captures ~14-16 bands, allowing selection of 12 closest to E_F")
    print("  Should include p-derived surface states near Fermi level")
    print()


if __name__ == '__main__':
    analyze_bands('tests/Bismuth_basis_40.out', num_wann=12)
