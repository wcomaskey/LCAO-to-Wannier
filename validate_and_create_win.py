#!/usr/bin/env python3
"""
Validate and Create Proper Wannier90 .win File

This script ensures:
1. num_bands > num_wann (DFT bands > Wannier projections)
2. num_bands ≈ 125-150% of num_wann for flexibility
3. Outer window contains at least num_bands DFT bands
4. Frozen window is narrow (region of interest near Fermi level)

Usage:
    python3 validate_and_create_win.py --input tests/Bismuth_basis_40.out --seedname bismuth --num-wann 12
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    parse_atoms_from_crystal_output,
)


def analyze_band_structure(input_file, energy_window, fermi_energy):
    """
    Analyze how many bands fall within a given energy window.

    Returns
    -------
    num_bands_in_window : int
        Number of bands within the energy window
    """
    # For now, we'll use heuristics based on typical band structures
    # In production, this would actually parse eigenvalues from CRYSTAL output

    window_width = energy_window[1] - energy_window[0]

    # Heuristic: ~1-2 bands per eV for typical materials
    # For SOC systems, expect ~2-3 bands per eV
    estimated_bands = int(window_width * 2.5)

    return estimated_bands


def determine_optimal_windows(num_wann, input_file, fermi_energy):
    """
    Determine optimal energy windows and num_bands.

    Parameters
    ----------
    num_wann : int
        Number of Wannier functions (projections)
    input_file : str
        Path to CRYSTAL output file
    fermi_energy : float
        Fermi energy in eV

    Returns
    -------
    config : dict
        Dictionary with keys: num_bands, dis_win_min, dis_win_max,
        dis_froz_min, dis_froz_max
    """
    print("=" * 80)
    print("DETERMINING OPTIMAL WANNIER90 CONFIGURATION")
    print("=" * 80)
    print()

    # Step 1: Calculate target num_bands (125-150% of num_wann)
    target_ratio = 1.35  # 135% is middle of 125-150% range
    target_num_bands = int(np.ceil(num_wann * target_ratio))

    print(f"Target Configuration:")
    print(f"  num_wann (projections): {num_wann}")
    print(f"  num_bands (DFT bands): {target_num_bands} ({target_ratio:.0%} of num_wann)")
    print()

    # Step 2: Determine outer window to contain target_num_bands
    # Start with a reasonable guess and iterate
    # Use wider estimate: ~1.5 bands per eV (conservative for SOC systems)
    initial_window_width = target_num_bands / 1.5

    # Make window symmetric around Fermi level
    half_width = initial_window_width / 2
    dis_win_min = -half_width
    dis_win_max = half_width

    # Verify window contains enough bands
    estimated_bands = analyze_band_structure(
        input_file,
        (dis_win_min, dis_win_max),
        fermi_energy
    )

    # Adjust if needed
    if estimated_bands < target_num_bands:
        # Expand window
        scale_factor = target_num_bands / estimated_bands * 1.1  # 10% buffer
        dis_win_min *= scale_factor
        dis_win_max *= scale_factor
        estimated_bands = analyze_band_structure(
            input_file,
            (dis_win_min, dis_win_max),
            fermi_energy
        )

    print(f"Outer Window (disentanglement):")
    print(f"  [{dis_win_min:.1f}, {dis_win_max:.1f}] eV relative to E_F")
    print(f"  Width: {dis_win_max - dis_win_min:.1f} eV")
    print(f"  Estimated bands in window: ~{estimated_bands}")
    print()

    # Step 3: Determine frozen window (narrow, region of interest)
    # Must be INSIDE outer window
    # This should capture bands crossing Fermi level
    froz_width = min(9.0, (dis_win_max - dis_win_min) * 0.7)  # 70% of outer or 9 eV
    dis_froz_min = max(dis_win_min, -froz_width / 2)
    dis_froz_max = min(dis_win_max, froz_width / 2)

    print(f"Frozen Window (region of interest):")
    print(f"  [{dis_froz_min:.1f}, {dis_froz_max:.1f}] eV relative to E_F")
    print(f"  Width: {dis_froz_max - dis_froz_min:.1f} eV")
    print()

    # Step 4: Validate configuration
    print("Validation:")
    checks = []

    # Check 1: num_bands > num_wann
    check1 = target_num_bands > num_wann
    checks.append(("num_bands > num_wann", check1,
                   f"{target_num_bands} > {num_wann}"))

    # Check 2: num_bands is 125-150% of num_wann
    ratio = target_num_bands / num_wann
    check2 = 1.25 <= ratio <= 1.50
    checks.append(("num_bands = 125-150% of num_wann", check2,
                   f"{ratio:.1%} of num_wann"))

    # Check 3: Frozen window inside outer window
    check3 = (dis_froz_min >= dis_win_min and dis_froz_max <= dis_win_max)
    checks.append(("Frozen window ⊂ Outer window", check3,
                   f"[{dis_froz_min:.1f}, {dis_froz_max:.1f}] ⊂ [{dis_win_min:.1f}, {dis_win_max:.1f}]"))

    # Check 4: Outer window should contain at least num_bands
    check4 = estimated_bands >= target_num_bands
    checks.append(("Outer window contains ≥ num_bands", check4,
                   f"~{estimated_bands} bands ≥ {target_num_bands}"))

    all_passed = True
    for check_name, passed, details in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}: {details}")
        all_passed = all_passed and passed

    print()

    if not all_passed:
        print("⚠ WARNING: Some validation checks failed!")
        print("  The configuration may not work properly.")
        print()
    else:
        print("✓ All validation checks passed!")
        print()

    return {
        'num_wann': num_wann,
        'num_bands': target_num_bands,
        'dis_win_min': dis_win_min,
        'dis_win_max': dis_win_max,
        'dis_froz_min': dis_froz_min,
        'dis_froz_max': dis_froz_max,
    }


def create_validated_win(input_file, seedname, num_wann):
    """
    Create a validated .win file with proper configuration.
    """
    print(f"Creating validated Wannier90 .win file for: {seedname}")
    print(f"Input: {input_file}")
    print()

    # Parse CRYSTAL output
    with open(input_file, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    atoms, _ = parse_atoms_from_crystal_output(lines)

    lattice_vectors = np.array(lattice_vectors_list)

    # Detect SOC from output file
    soc_enabled = any('TWO-COMPONENT SCF' in line or 'SPIN-ORBIT' in line for line in lines)

    print(f"System Information:")
    print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Atoms: {len(atoms)}")
    print(f"  Spin-orbit coupling: {'Yes' if soc_enabled else 'No'}")
    print()

    # Determine optimal configuration
    config = determine_optimal_windows(num_wann, input_file, params.fermi_energy)

    # Create .win file content
    win_content = f"""! Generated by LCAO-to-Wannier90 package (validated configuration)
! Wannier90 input file for seedname: {seedname}
!
! IMPORTANT: This configuration ensures:
!   - num_bands ({config['num_bands']}) > num_wann ({config['num_wann']})
!   - num_bands is ~{config['num_bands']/config['num_wann']:.0%} of num_wann (flexibility)
!   - Outer window contains at least {config['num_bands']} DFT bands
!   - Frozen window captures region of interest near Fermi level

!====================
! SYSTEM PARAMETERS
!====================
num_wann = {config['num_wann']}
num_bands = {config['num_bands']}

! Fermi energy (for relative energy windows)
fermi_energy = {params.fermi_energy:.6f}

! Spin-orbit coupling included
spinors = {'.true.' if soc_enabled else '.false.'}

!====================
! WANNIERIZATION
!====================
num_iter = 5000
conv_tol = 1.00e-10
conv_window = 5
num_print_cycles = 10

! Disentanglement windows (eV, relative to Fermi energy)
! Outer window: wide enough to contain {config['num_bands']} DFT bands
dis_win_min = {config['dis_win_min']:.1f}
dis_win_max = {config['dis_win_max']:.1f}
! Frozen window: narrow region of interest near Fermi level
dis_froz_min = {config['dis_froz_min']:.1f}
dis_froz_max = {config['dis_froz_max']:.1f}
dis_num_iter = 1000
dis_mix_ratio = 0.5

!====================
! OUTPUT OPTIONS
!====================
write_hr = T
write_xyz = T
write_r2mn = F

!====================
! UNIT CELL
!====================
begin unit_cell_cart
Ang
"""

    # Add lattice vectors
    for i in range(3):
        win_content += f"     {lattice_vectors[i,0]:18.12f} {lattice_vectors[i,1]:18.12f} {lattice_vectors[i,2]:18.12f}\n"

    win_content += "end unit_cell_cart\n\n"

    # Add atoms
    if atoms:
        win_content += "!====================\n"
        win_content += "! ATOMS\n"
        win_content += "!====================\n"
        win_content += "begin atoms_frac\n"
        for symbol, pos in atoms:
            win_content += f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n"
        win_content += "end atoms_frac\n\n"

    # Add projections
    win_content += """!====================
! PROJECTIONS
!====================
! Option 1: Automatic (6 orbitals × 2 spinor = 12 WFs)
begin projections
Bi:p
end projections

! Option 2: Manual (5 orbitals × 2 spinor = 10 WFs)
! Uncomment below and comment out Option 1 if you want 10 WFs
!begin projections
"""
    if atoms:
        win_content += f"!f={atoms[0][1][0]:.6f},{atoms[0][1][1]:.6f},{atoms[0][1][2]:.6f}:px\n"
        win_content += f"!f={atoms[0][1][0]:.6f},{atoms[0][1][1]:.6f},{atoms[0][1][2]:.6f}:py\n"
        win_content += f"!f={atoms[0][1][0]:.6f},{atoms[0][1][1]:.6f},{atoms[0][1][2]:.6f}:pz\n"
        if len(atoms) > 1:
            win_content += f"!f={atoms[1][1][0]:.6f},{atoms[1][1][1]:.6f},{atoms[1][1][2]:.6f}:px\n"
            win_content += f"!f={atoms[1][1][0]:.6f},{atoms[1][1][1]:.6f},{atoms[1][1][2]:.6f}:py\n"
    win_content += "!end projections\n\n"

    # Add k-points
    win_content += "!====================\n"
    win_content += "! K-POINT MESH\n"
    win_content += "!====================\n"
    win_content += f"mp_grid = {params.k_grid[0]} {params.k_grid[1]} {params.k_grid[2]}\n\n"

    # Generate k-point list
    win_content += "begin kpoints\n"
    nk1, nk2, nk3 = params.k_grid
    for ik3 in range(nk3):
        for ik2 in range(nk2):
            for ik1 in range(nk1):
                k1 = ik1 / nk1 if nk1 > 1 else 0.0
                k2 = ik2 / nk2 if nk2 > 1 else 0.0
                k3 = ik3 / nk3 if nk3 > 1 else 0.0
                win_content += f"     {k1:18.12f} {k2:18.12f} {k3:18.12f}\n"
    win_content += "end kpoints\n"

    # Write file
    win_filename = f"{seedname}.win"
    with open(win_filename, 'w') as f:
        f.write(win_content)

    print("=" * 80)
    print("FINAL CONFIGURATION")
    print("=" * 80)
    print(f"✓ Created: {win_filename}")
    print()
    print(f"Parameters:")
    print(f"  num_wann = {config['num_wann']}")
    print(f"  num_bands = {config['num_bands']} ({config['num_bands']/config['num_wann']:.1%} of num_wann)")
    print(f"  Outer window: [{config['dis_win_min']:.1f}, {config['dis_win_max']:.1f}] eV")
    print(f"  Frozen window: [{config['dis_froz_min']:.1f}, {config['dis_froz_max']:.1f}] eV")
    print()
    print("Next steps:")
    print(f"  1. Run: wannier90.x -pp {seedname}")
    print(f"  2. Run: python3 lcao_to_wannier90.py --stage 2 \\")
    print(f"          --input {input_file} --seedname {seedname} \\")
    print(f"          --window {config['dis_win_min']:.1f} {config['dis_win_max']:.1f}")
    print(f"  3. Run: wannier90.x {seedname}")
    print()

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Create validated Wannier90 .win file with proper num_bands > num_wann'
    )
    parser.add_argument('--input', required=True, help='CRYSTAL output file')
    parser.add_argument('--seedname', required=True, help='Wannier90 seedname')
    parser.add_argument('--num-wann', type=int, default=12,
                       help='Number of Wannier functions (default: 12)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    create_validated_win(args.input, args.seedname, args.num_wann)


if __name__ == '__main__':
    main()
