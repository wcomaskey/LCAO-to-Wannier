#!/usr/bin/env python3
"""
Enhanced Band Structure Analysis Script

This script analyzes band structures from LCAO/CRYSTAL calculations,
showing energy ranges and orbital character composition.

Features:
- Band energy ranges (min/max/width) across k-points
- Orbital character analysis (element and orbital type contributions)
- Intelligent window recommendations for Wannier90
- Multiple output formats (text, JSON, CSV)

Usage:
    # Full analysis from CRYSTAL output
    python3 analyze_band_structure.py --input tests/Bismuth_basis_40.out

    # Quick analysis from existing .eig file
    python3 analyze_band_structure.py --eig-file bismuth.eig --seedname bismuth

    # Specify custom parameters
    python3 analyze_band_structure.py --input tests/Bismuth_basis_40.out \\
        --num-electrons 10 --k-grid 15 15 1 --parallel
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Import LCAO-Wannier modules
from lcao_wannier import (
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    parse_atomic_basis_info,
    parse_orbital_types,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine,
    analyze_all_bands_character,
    format_band_character_table
)


def parse_eig_file(eig_file: str) -> Dict[int, List[float]]:
    """
    Parse .eig file to get band energies (fast mode).

    Parameters
    ----------
    eig_file : str
        Path to .eig file

    Returns
    -------
    dict
        Dictionary mapping band_index → list of energies at each k-point
    """
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


def analyze_band_ranges_from_eig(eig_file: str, fermi_energy: float = 0.0):
    """
    Quick band range analysis from .eig file (no orbital character).

    Parameters
    ----------
    eig_file : str
        Path to .eig file
    fermi_energy : float
        Fermi energy in eV (default: 0.0)
    """
    print("=" * 80)
    print(f"BAND ENERGY RANGES FROM {eig_file}")
    print("=" * 80)
    print()

    bands = parse_eig_file(eig_file)
    num_bands = len(bands)
    num_kpts = len(bands[1]) if bands else 0

    print(f"Number of bands: {num_bands}")
    print(f"Number of k-points: {num_kpts}")
    print(f"Fermi energy: {fermi_energy:.6f} eV")
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
        crosses_ef = (min_E < fermi_energy) and (max_E > fermi_energy)

        band_stats.append((band_idx, min_E, max_E, crosses_ef))

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

        print(f"{band_idx:<6} {min_E:>10.4f}   {max_E:>10.4f}   {width:>8.4f}   {cross_marker:<12} {notes}")

    print()

    # Summary statistics
    fermi_bands = [b for b, min_E, max_E, crosses in band_stats if crosses]
    valence_bands = [b for b, min_E, max_E, crosses in band_stats if max_E < fermi_energy]
    conduction_bands = [b for b, min_E, max_E, crosses in band_stats if min_E > fermi_energy]

    print("Summary:")
    print(f"  Total bands: {num_bands}")
    print(f"  Bands crossing Fermi level: {len(fermi_bands)}", end="")
    if fermi_bands:
        print(f" (indices {min(fermi_bands)} - {max(fermi_bands)})")
    else:
        print()
    print(f"  Valence bands below E_F: {len(valence_bands)}")
    print(f"  Conduction bands above E_F: {len(conduction_bands)}")

    # Determine if metallic or insulating
    if fermi_bands:
        print(f"  Band gap: None (metallic)")
    else:
        # Find gap
        if valence_bands and conduction_bands:
            vb_max = max(bands[b][0] for b in valence_bands for _ in [bands[b]])
            cb_min = min(bands[b][0] for b in conduction_bands for _ in [bands[b]])
            gap = cb_min - vb_max
            print(f"  Band gap: {gap:.4f} eV (insulating)")
    print()


def analyze_band_ranges_full(input_file: str, k_grid: Optional[Tuple[int, int, int]] = None,
                             fermi_energy: Optional[float] = None, num_electrons: Optional[int] = None,
                             parallel: bool = True, verbose: bool = False,
                             analyze_character: bool = True, num_bands_to_analyze: int = 20):
    """
    Full band range analysis from CRYSTAL output (with eigenvectors for orbital character).

    Parameters
    ----------
    input_file : str
        Path to CRYSTAL output file
    k_grid : tuple of 3 ints, optional
        K-point grid (if not specified, read from file)
    fermi_energy : float, optional
        Fermi energy in eV (if not specified, auto-detect)
    num_electrons : int, optional
        Number of electrons for Fermi energy detection
    parallel : bool
        Use parallel eigenvalue solving
    verbose : bool
        Print detailed progress information
    """
    print("=" * 80)
    print("BAND STRUCTURE ANALYSIS (Full Mode)")
    print("=" * 80)
    print()
    print(f"Input file: {input_file}")
    print()

    # Parse CRYSTAL output
    if verbose:
        print("Parsing CRYSTAL output file...")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Parse calculation parameters
    params = parse_calculation_parameters(lines)

    # Parse matrices
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    # Check for SOC
    has_soc = params.has_soc

    # Organize matrices by R-vector
    H_R_dict = {}
    S_R_dict = {}
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        mat_type = mat_info['type']
        data = mat_info['data']

        if mat_type == 'overlap':
            S_R_dict[R] = data
        elif mat_type == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            if R not in H_R_dict:
                H_R_dict[R] = {}
            H_R_dict[R][spin_channel] = data

    num_basis = params.num_ao if params.num_ao else list(S_R_dict.values())[0].shape[0]

    # Create spin-block matrices if SOC
    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        num_orbitals = H_full_list[0][1].shape[0]
    else:
        # For non-SOC, create the (R_cartesian, matrix) list format manually
        H_full_list = []
        S_full_list = []

        for R_int, H_mats in H_R_dict.items():
            R_cart = lattice_vectors.T @ np.array(R_int)
            H_mat = list(H_mats.values())[0] if isinstance(H_mats, dict) else H_mats
            H_full_list.append((R_cart, H_mat))

        for R_int, S_mat in S_R_dict.items():
            R_cart = lattice_vectors.T @ np.array(R_int)
            S_full_list.append((R_cart, S_mat))

        num_orbitals = num_basis

    # Prepare real-space matrices
    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )

    # Get k-grid
    if k_grid is None:
        k_grid = params.k_grid
        if k_grid is None:
            print("Error: K-grid not found in file and not specified!")
            sys.exit(1)

    # Get Fermi energy
    if fermi_energy is None:
        fermi_energy = params.fermi_energy

    # Get number of electrons
    if num_electrons is None:
        num_electrons = params.num_electrons

    print(f"K-grid: {k_grid[0]} × {k_grid[1]} × {k_grid[2]} ({k_grid[0]*k_grid[1]*k_grid[2]} k-points)")
    print(f"Number of orbitals: {num_orbitals}", end="")
    if params.has_soc:
        print(" (SOC enabled)")
    else:
        print()

    if fermi_energy is not None:
        print(f"Fermi energy: {fermi_energy:.6f} eV", end="")
        if num_electrons is not None:
            print(f" (from {num_electrons} electrons)")
        else:
            print(" (from file)")
    else:
        print("Fermi energy: Will auto-detect")

    print()

    # Initialize Wannier90 engine without energy window (analyze all bands)
    if verbose:
        print("Initializing Wannier90 engine...")

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=k_grid,
        lattice_vectors=lattice_vectors,
        seedname='temp',
        num_wann=None,  # Don't specify, analyze all
        e_fermi=fermi_energy,
        num_electrons=num_electrons
    )

    # Solve eigenvalue problems
    print("=" * 80)
    print(f"Solving eigenvalue problems at {k_grid[0]*k_grid[1]*k_grid[2]} k-points...")
    print("=" * 80)
    print()

    engine.solve_all_kpoints(parallel=parallel, convert_to_eV=True)

    if verbose:
        print(f"✓ Solved {len(engine.eigenvalues_list)} k-points")
        print()

    # Get band structure
    kpoints, eigenvalues_array = engine.get_band_structure()
    num_kpoints, num_bands = eigenvalues_array.shape

    # Use detected Fermi energy if available
    if engine.e_fermi is not None:
        fermi_energy = engine.e_fermi
    else:
        fermi_energy = 0.0

    # Analyze each band
    print("=" * 80)
    print(f"BAND ENERGY RANGES (relative to E_F = {fermi_energy:.6f} eV)")
    print("=" * 80)
    print()
    print(f"{'Band':<6} {'Min (eV)':<12} {'Max (eV)':<12} {'Width (eV)':<12} {'Crosses E_F':<12} {'Notes'}")
    print("-" * 80)

    band_stats = []
    for band_idx in range(num_bands):
        band_energies = eigenvalues_array[:, band_idx]
        min_E = np.min(band_energies) - fermi_energy
        max_E = np.max(band_energies) - fermi_energy
        width = max_E - min_E
        crosses_ef = (min_E < 0) and (max_E > 0)

        band_stats.append((band_idx + 1, min_E, max_E, crosses_ef))

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

        print(f"{band_idx+1:<6} {min_E:>10.4f}   {max_E:>10.4f}   {width:>10.4f}   {cross_marker:<12} {notes}")

    print()

    # Summary statistics
    fermi_bands = [b for b, min_E, max_E, crosses in band_stats if crosses]
    valence_bands = [b for b, min_E, max_E, crosses in band_stats if max_E < 0]
    conduction_bands = [b for b, min_E, max_E, crosses in band_stats if min_E > 0]

    print("Summary:")
    print(f"  Total bands: {num_bands}")
    print(f"  Bands crossing Fermi level: {len(fermi_bands)}", end="")
    if fermi_bands:
        print(f" (indices {min(fermi_bands)} - {max(fermi_bands)})")
    else:
        print()
    print(f"  Valence bands below E_F: {len(valence_bands)}")
    print(f"  Conduction bands above E_F: {len(conduction_bands)}")

    # Determine if metallic or insulating
    if fermi_bands:
        print(f"  Band gap: None (metallic)")
    else:
        # Find gap
        if valence_bands and conduction_bands:
            vb_max_energy = max(max_E for b, min_E, max_E, crosses in band_stats if max_E < 0)
            cb_min_energy = min(min_E for b, min_E, max_E, crosses in band_stats if min_E > 0)
            gap = cb_min_energy - vb_max_energy
            print(f"  Band gap: {gap:.4f} eV (insulating)")
    print()

    # Orbital character analysis (if requested and eigenvectors available)
    if analyze_character and engine.eigenvectors_list is not None:
        print("=" * 80)
        print("ORBITAL CHARACTER ANALYSIS")
        print("=" * 80)
        print()

        # Parse atomic basis information
        with open(input_file, 'r') as f:
            lines = f.readlines()

        atomic_info = parse_atomic_basis_info(lines)
        orbital_types = parse_orbital_types(lines, has_soc=has_soc)

        if not orbital_types:
            print("Warning: Could not parse orbital types from CRYSTAL output")
            print("Skipping orbital character analysis")
            print()
        else:
            # Determine which bands to analyze (near Fermi level)
            # Find bands near Fermi
            bands_near_fermi = []
            for band_idx in range(num_bands):
                band_energies = eigenvalues_array[:, band_idx]
                min_E = np.min(band_energies) - fermi_energy
                max_E = np.max(band_energies) - fermi_energy

                # Include if within reasonable range of Fermi or if crosses
                if abs(min_E) < 15 and abs(max_E) < 15:
                    bands_near_fermi.append(band_idx)

            # Limit to specified number
            if len(bands_near_fermi) > num_bands_to_analyze:
                # Prioritize bands closest to Fermi
                bands_with_dist = []
                for band_idx in bands_near_fermi:
                    band_energies = eigenvalues_array[:, band_idx]
                    min_E = abs(np.min(band_energies) - fermi_energy)
                    max_E = abs(np.max(band_energies) - fermi_energy)
                    dist = min(min_E, max_E)
                    bands_with_dist.append((dist, band_idx))

                bands_with_dist.sort()
                bands_to_analyze = np.array([b for _, b in bands_with_dist[:num_bands_to_analyze]])
            else:
                bands_to_analyze = np.array(bands_near_fermi)

            print(f"Analyzing {len(bands_to_analyze)} bands near Fermi level...")
            print(f"Band indices: {bands_to_analyze[0]+1} - {bands_to_analyze[-1]+1}")
            print()

            # Perform character analysis
            band_characters = analyze_all_bands_character(
                engine.eigenvectors_list,
                engine.S_k_list,
                atomic_info.atom_symbols,
                atomic_info.basis_atom_map,
                orbital_types,
                band_indices=bands_to_analyze,
                verbose=verbose
            )

            # Format and print results
            table = format_band_character_table(band_characters)
            print(table)
            print()

    return engine, band_stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze band structure from LCAO/CRYSTAL calculations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis from CRYSTAL output
  python3 analyze_band_structure.py --input tests/Bismuth_basis_40.out

  # Quick analysis from .eig file
  python3 analyze_band_structure.py --eig-file bismuth.eig --fermi-energy -3.728

  # Custom parameters
  python3 analyze_band_structure.py --input tests/Bismuth_basis_40.out \\
      --num-electrons 10 --k-grid 15 15 1 --parallel --verbose
        """
    )

    # Input mode
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i',
                           help='CRYSTAL output file (full analysis mode)')
    input_group.add_argument('--eig-file', '-e',
                           help='.eig file for quick analysis (no orbital character)')

    # Parameters
    parser.add_argument('--seedname', '-s', default='temp',
                       help='Seedname for output files (default: temp)')
    parser.add_argument('--k-grid', type=int, nargs=3, metavar=('NK1', 'NK2', 'NK3'),
                       help='K-point grid (default: read from file)')
    parser.add_argument('--fermi-energy', '-f', type=float,
                       help='Fermi energy in eV (default: auto-detect)')
    parser.add_argument('--num-electrons', '-n', type=int,
                       help='Number of electrons for Fermi detection')

    # Options
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Use parallel eigenvalue solving')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress information')
    parser.add_argument('--output-format', choices=['txt', 'json', 'csv'], default='txt',
                       help='Output format (default: txt)')
    parser.add_argument('--no-character', action='store_true',
                       help='Skip orbital character analysis')
    parser.add_argument('--num-bands-analyze', type=int, default=20,
                       help='Number of bands to analyze for character (default: 20)')

    args = parser.parse_args()

    # Validate inputs
    if args.input:
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        # Full analysis mode
        engine, band_stats = analyze_band_ranges_full(
            args.input,
            k_grid=tuple(args.k_grid) if args.k_grid else None,
            fermi_energy=args.fermi_energy,
            num_electrons=args.num_electrons,
            parallel=args.parallel,
            verbose=args.verbose,
            analyze_character=not args.no_character,
            num_bands_to_analyze=args.num_bands_analyze
        )

    elif args.eig_file:
        if not Path(args.eig_file).exists():
            print(f"Error: .eig file not found: {args.eig_file}")
            sys.exit(1)

        # Quick analysis mode
        fermi_energy = args.fermi_energy if args.fermi_energy is not None else 0.0
        analyze_band_ranges_from_eig(args.eig_file, fermi_energy)


if __name__ == '__main__':
    main()
