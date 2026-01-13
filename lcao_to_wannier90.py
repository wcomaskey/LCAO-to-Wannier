#!/usr/bin/env python3
"""
LCAO-to-Wannier90 Two-Stage Workflow Script

This script implements the correct Wannier90 workflow with two stages:

Stage 1: Generate .win file from CRYSTAL/LCAO output
Stage 2: Generate .eig, .amn, .mmn files using .nnkp neighbor information

Usage:
    # Stage 1: Create .win file
    python lcao_to_wannier90.py --stage 1 --input material.out --seedname material

    # Then run: wannier90.x -pp material

    # Stage 2: Create data files using .nnkp
    python lcao_to_wannier90.py --stage 2 --input material.out --seedname material

    # Then run: wannier90.x material

For help:
    python lcao_to_wannier90.py --help
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

from lcao_wannier import (
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    parse_atomic_basis_info,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine,
    suggest_optimal_window,
    analyze_band_window,
    parse_atoms_from_crystal_output,
    estimate_fermi_energy,
)


def stage1_create_win(args):
    """
    Stage 1: Parse LCAO output and create .win file only.

    This prepares the Wannier90 input file so the user can run
    'wannier90.x -pp seedname' to generate the .nnkp file.
    """
    print("=" * 80)
    print("STAGE 1: Creating Wannier90 Parameter File (.win)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Parse CRYSTAL output
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed calculation parameters:")
    if params.fermi_energy is not None:
        print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    else:
        print(f"  Fermi energy: Not found (will estimate later)")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Number of AOs: {params.num_ao}")

    # Use SOC detection from parser (TWO-COMPONENT SCF marker)
    has_soc = params.has_soc

    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # If SOC is detected, we need to double the number of orbitals for spinors
    if has_soc:
        print(f"  → Doubling orbitals for spinors: {params.num_ao} → {params.num_ao * 2}")

    # Organize matrices by R-vector
    print("\nStep 2: Organizing matrices...")
    print("-" * 80)

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
    print(f"✓ Organized matrices")
    print(f"  Basis size: {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Create spin-block matrices if SOC
    print("\nStep 3: Creating matrices...")
    print("-" * 80)

    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        matrix_size = H_full_list[0][1].shape[0]
        print(f"✓ Created {matrix_size}×{matrix_size} SOC matrices")
    else:
        # For non-SOC, create the (R_cartesian, matrix) list format manually
        H_full_list = []
        S_full_list = []

        for R_int, H_mats in H_R_dict.items():
            # Convert integer R to Cartesian
            R_cart = lattice_vectors.T @ np.array(R_int)
            # Use spin channel 0 (or average if multiple)
            H_mat = list(H_mats.values())[0] if isinstance(H_mats, dict) else H_mats
            H_full_list.append((R_cart, H_mat))

        for R_int, S_mat in S_R_dict.items():
            R_cart = lattice_vectors.T @ np.array(R_int)
            S_full_list.append((R_cart, S_mat))

        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} matrices (no SOC)")

    # Prepare real-space matrices in the format engine expects
    print("\nStep 4: Preparing real-space matrices...")
    print("-" * 80)

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    print(f"✓ Prepared {len(real_space_matrices)} R-vectors")

    # Determine energy window
    print("\nStep 5: Determining energy window...")
    print("-" * 80)

    if args.window:
        e_min, e_max = args.window
        print(f"Using user-specified window: [{e_min:.2f}, {e_max:.2f}] eV")
    else:
        e_min, e_max = -5.0, 3.0  # Default window
        print(f"Using default window: [{e_min:.2f}, {e_max:.2f}] eV (relative to E_F)")

    # Initialize engine
    print("\nStep 6: Initializing Wannier90 engine...")
    print("-" * 80)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=args.seedname,
        num_wann=None,  # Will be determined from window
        outer_window=(e_min, e_max),
        e_fermi=params.fermi_energy,
        window_is_relative=True
    )

    print("✓ Engine initialized")

    # Parse atomic basis information for phase-corrected MMN
    print("\nParsing atomic basis information...")
    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
        atomic_info = parse_atomic_basis_info(lines)
        engine.atom_positions = atomic_info.atom_positions

        # For SOC systems, double the basis_atom_map (spin up + spin down)
        num_orbitals = engine.num_orbitals
        if num_orbitals == 2 * atomic_info.num_basis:
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,
                atomic_info.basis_atom_map
            ])
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions (SOC: doubled basis map)")
        else:
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction")

    # Solve eigenvalue problems
    print("\nStep 7: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel)
    print("✓ Eigenvalue problems solved")

    # Estimate Fermi energy if needed
    if params.fermi_energy is None:
        print("\nEstimating Fermi energy...")
        if params.num_electrons is not None:
            fermi_energy = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=params.num_electrons,
                method='auto'
            )
            engine.e_fermi = fermi_energy
            print(f"✓ Estimated Fermi energy: {fermi_energy:.6f} eV")
        else:
            print("⚠ Cannot estimate Fermi energy (num_electrons not found)")
            print("  Using E_F = 0.0 eV")
            engine.e_fermi = 0.0

    # Analyze bands and select window
    print("\nStep 8: Analyzing band structure and selecting bands...")
    print("-" * 80)

    result = analyze_band_window(
        engine.eigenvalues_list,
        outer_window=(e_min, e_max),
        e_fermi=engine.e_fermi,
        window_is_relative=True
    )

    # Set num_wann based on frozen bands
    num_frozen = result.num_wann
    if num_frozen == 0:
        print("ERROR: No bands found in the energy window!")
        print("Please adjust the energy window and try again.")
        sys.exit(1)

    engine.num_wann = num_frozen
    engine.selected_band_indices = result.frozen_indices

    print(f"✓ Selected {num_frozen} bands for Wannier functions")
    print(f"  Energy range: [{result.frozen_energy_range[0]:.2f}, {result.frozen_energy_range[1]:.2f}] eV")

    # Select projection orbitals
    print("\nStep 9: Selecting projection orbitals...")
    print("-" * 80)

    engine.select_projections(verbose=True)

    # Parse atoms from CRYSTAL output if available
    print("\nStep 10: Extracting atomic positions...")
    print("-" * 80)

    try:
        atoms = parse_atoms_from_crystal_output(args.input)
        print(f"✓ Found {len(atoms)} atoms")
        for symbol, pos in atoms:
            print(f"  {symbol}: {pos}")
    except Exception as e:
        print(f"⚠ Could not parse atoms: {e}")
        print("  Will create .win without atoms block")
        atoms = None

    # Write .win file only
    print(f"\nStep 11: Writing {args.seedname}.win file...")
    print("-" * 80)

    # Prepare projections
    projections = args.projections if args.projections else None

    # Write only the .win file
    engine.write_files(
        verbose=True,
        write_win=True,
        use_nnkp=False,  # Don't use .nnkp in stage 1
        atoms=atoms,
        projections=projections,
        spinors=has_soc,
        bands_plot=args.bands_plot,
        kpoint_path=None  # Could add option for this
    )

    # Delete the .eig, .amn, .mmn files if they were created
    # (write_files creates them by default, but we only want .win in stage 1)
    for ext in ['.eig', '.amn', '.mmn']:
        filepath = f"{args.seedname}{ext}"
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  (Removed premature {filepath})")

    print()
    print("=" * 80)
    print("STAGE 1 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.win")
    print()
    print("NEXT STEP:")
    print(f"  Run Wannier90 preprocessing to generate neighbor information:")
    print(f"  → wannier90.x -pp {args.seedname}")
    print()
    print(f"  This will create {args.seedname}.nnkp")
    print()
    print("Then proceed to Stage 2:")
    print(f"  python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
    print("=" * 80)


def stage2_create_data_files(args):
    """
    Stage 2: Read .nnkp and create .eig, .amn, .mmn files.

    This uses the neighbor information from Wannier90's preprocessing
    to generate data files with the exact neighbor structure expected.
    """
    print("=" * 80)
    print("STAGE 2: Creating Wannier90 Data Files (.eig, .amn, .mmn)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check that .nnkp file exists
    nnkp_file = f"{args.seedname}.nnkp"
    if not os.path.exists(nnkp_file):
        print(f"ERROR: {nnkp_file} not found!")
        print()
        print("You must run Stage 1 first, then run Wannier90 preprocessing:")
        print(f"  1. python {sys.argv[0]} --stage 1 --input {args.input} --seedname {args.seedname}")
        print(f"  2. wannier90.x -pp {args.seedname}")
        print(f"  3. python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
        print()
        sys.exit(1)

    # Check that .win file exists
    win_file = f"{args.seedname}.win"
    if not os.path.exists(win_file):
        print(f"ERROR: {win_file} not found!")
        print("Please run Stage 1 first to create the .win file.")
        sys.exit(1)

    print(f"✓ Found {nnkp_file}")
    print(f"✓ Found {win_file}")
    print()

    # Parse CRYSTAL output (same as stage 1)
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed calculation parameters:")
    if params.fermi_energy is not None:
        print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    else:
        print(f"  Fermi energy: Not found (will estimate later)")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Number of AOs: {params.num_ao}")

    # Use SOC detection from parser (TWO-COMPONENT SCF marker)
    has_soc = params.has_soc

    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # If SOC is detected, we need to double the number of orbitals for spinors
    if has_soc:
        print(f"  → Doubling orbitals for spinors: {params.num_ao} → {params.num_ao * 2}")

    # Organize matrices by R-vector
    print("\nStep 2: Organizing matrices...")
    print("-" * 80)

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
    print(f"✓ Organized matrices")
    print(f"  Basis size: {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Create spin-block matrices if SOC
    print("\nStep 3: Creating matrices...")
    print("-" * 80)

    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        matrix_size = H_full_list[0][1].shape[0]
        print(f"✓ Created {matrix_size}×{matrix_size} SOC matrices")
    else:
        # For non-SOC, create the (R_cartesian, matrix) list format manually
        H_full_list = []
        S_full_list = []

        for R_int, H_mats in H_R_dict.items():
            # Convert integer R to Cartesian
            R_cart = lattice_vectors.T @ np.array(R_int)
            # Use spin channel 0 (or average if multiple)
            H_mat = list(H_mats.values())[0] if isinstance(H_mats, dict) else H_mats
            H_full_list.append((R_cart, H_mat))

        for R_int, S_mat in S_R_dict.items():
            R_cart = lattice_vectors.T @ np.array(R_int)
            S_full_list.append((R_cart, S_mat))

        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} matrices (no SOC)")

    # Prepare real-space matrices
    print("\nStep 4: Preparing real-space matrices...")
    print("-" * 80)

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    print(f"✓ Prepared {len(real_space_matrices)} R-vectors")

    # Read num_wann and num_bands from .win file if it exists
    print("\nStep 5: Reading parameters from .win file...")
    print("-" * 80)

    num_wann_from_win = None
    num_bands_from_win = None
    dis_froz_min_from_win = None
    dis_froz_max_from_win = None
    win_file = f"{args.seedname}.win"

    if os.path.exists(win_file):
        with open(win_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('num_wann'):
                    try:
                        num_wann_from_win = int(line.split('=')[1].strip())
                        print(f"  Found num_wann = {num_wann_from_win}")
                    except:
                        pass
                elif line.startswith('num_bands'):
                    try:
                        num_bands_from_win = int(line.split('=')[1].strip())
                        print(f"  Found num_bands = {num_bands_from_win}")
                    except:
                        pass
                elif line.startswith('dis_froz_min'):
                    try:
                        dis_froz_min_from_win = float(line.split('=')[1].strip())
                        print(f"  Found dis_froz_min = {dis_froz_min_from_win}")
                    except:
                        pass
                elif line.startswith('dis_froz_max'):
                    try:
                        dis_froz_max_from_win = float(line.split('=')[1].strip())
                        print(f"  Found dis_froz_max = {dis_froz_max_from_win}")
                    except:
                        pass
        print(f"✓ Read parameters from {win_file}")
    else:
        print(f"⚠ Warning: {win_file} not found, using automatic band selection")

    # Determine energy window (same as stage 1)
    print("\nStep 6: Determining energy window...")
    print("-" * 80)

    if args.window:
        e_min, e_max = args.window
        print(f"Using user-specified window: [{e_min:.2f}, {e_max:.2f}] eV")
    else:
        e_min, e_max = -5.0, 3.0
        print(f"Using default window: [{e_min:.2f}, {e_max:.2f}] eV (relative to E_F)")

    # Initialize engine
    print("\nStep 7: Initializing Wannier90 engine...")
    print("-" * 80)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=args.seedname,
        num_wann=num_wann_from_win,  # Use value from .win file
        outer_window=(e_min, e_max),
        e_fermi=params.fermi_energy,
        window_is_relative=True
    )

    print("✓ Engine initialized")

    # Parse atomic basis information for phase-corrected MMN
    print("\nParsing atomic basis information...")
    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
        atomic_info = parse_atomic_basis_info(lines)
        engine.atom_positions = atomic_info.atom_positions

        # For SOC systems, double the basis_atom_map (spin up + spin down)
        num_orbitals = engine.num_orbitals
        if num_orbitals == 2 * atomic_info.num_basis:
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,
                atomic_info.basis_atom_map
            ])
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions (SOC: doubled basis map)")
        else:
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction (may cause negative spreads!)")

    # Solve eigenvalue problems
    print("\nStep 7: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel)
    print("✓ Eigenvalue problems solved")

    # Estimate Fermi energy if needed
    if params.fermi_energy is None:
        print("\nEstimating Fermi energy...")
        if params.num_electrons is not None:
            fermi_energy = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=params.num_electrons,
                method='auto'
            )
            engine.e_fermi = fermi_energy
            print(f"✓ Estimated Fermi energy: {fermi_energy:.6f} eV")
        else:
            print("⚠ Cannot estimate Fermi energy (num_electrons not found)")
            print("  Using E_F = 0.0 eV")
            engine.e_fermi = 0.0

    # Analyze bands or use values from .win file
    print("\nStep 8: Analyzing band structure and selecting bands...")
    print("-" * 80)

    if num_bands_from_win is not None and num_wann_from_win is not None:
        # Use values from .win file
        print(f"Using num_wann = {num_wann_from_win} and num_bands = {num_bands_from_win} from .win file")

        result = analyze_band_window(
            engine.eigenvalues_list,
            outer_window=(e_min, e_max),
            e_fermi=engine.e_fermi,
            window_is_relative=True
        )

        # Combine frozen and partial bands for outer window
        outer_bands = np.concatenate([result.frozen_indices, result.partial_indices])
        outer_bands = np.sort(outer_bands)

        if len(outer_bands) < num_bands_from_win:
            print(f"⚠ Warning: Energy window contains only {len(outer_bands)} bands")
            print(f"           but .win file specifies num_bands = {num_bands_from_win}")
            print(f"  Using all {len(outer_bands)} bands in window")
            engine.selected_band_indices = list(outer_bands)
        else:
            # Select first num_bands from the outer window
            engine.selected_band_indices = list(outer_bands[:num_bands_from_win])

        engine.num_wann = num_wann_from_win
        print(f"✓ Selected {len(engine.selected_band_indices)} bands in energy window")
        print(f"  Band indices: {engine.selected_band_indices[0]}-{engine.selected_band_indices[-1]} (1-based: {engine.selected_band_indices[0]+1}-{engine.selected_band_indices[-1]+1})")
    else:
        # Use automatic band selection
        result = analyze_band_window(
            engine.eigenvalues_list,
            outer_window=(e_min, e_max),
            e_fermi=engine.e_fermi,
            window_is_relative=True
        )

        num_frozen = result.num_wann
        if num_frozen == 0:
            print("ERROR: No bands found in the energy window!")
            sys.exit(1)

        engine.num_wann = num_frozen
        engine.selected_band_indices = result.frozen_indices

        print(f"✓ Selected {num_frozen} bands for Wannier functions")

    # Validate band selection
    print("\nValidating band selection...")
    print("-" * 80)

    selected = np.array(engine.selected_band_indices)

    # Check 1: Are bands contiguous?
    gaps = np.diff(selected)
    if np.any(gaps > 1):
        gap_locations = np.where(gaps > 1)[0]
        print(f"⚠ Warning: Selected bands are non-contiguous!")
        print(f"  Gaps at band indices: {[selected[i] for i in gap_locations]}")
    else:
        print(f"✓ Selected bands are contiguous")

    # Check 2: Do selected bands span Fermi level?
    selected_e_min = min([result.band_ranges[i][0] for i in selected])
    selected_e_max = max([result.band_ranges[i][1] for i in selected])

    if engine.e_fermi < selected_e_min:
        print(f"⚠ Warning: Fermi level ({engine.e_fermi:.2f} eV) is BELOW all selected bands")
        print(f"  Selected range: [{selected_e_min:.2f}, {selected_e_max:.2f}] eV")
        print(f"  You may have selected only conduction bands!")
    elif engine.e_fermi > selected_e_max:
        print(f"⚠ Warning: Fermi level ({engine.e_fermi:.2f} eV) is ABOVE all selected bands")
        print(f"  Selected range: [{selected_e_min:.2f}, {selected_e_max:.2f}] eV")
        print(f"  You may have selected only core/valence bands!")
    else:
        # Count bands below and above E_F
        bands_below = sum(1 for i in selected if result.band_ranges[i][1] < engine.e_fermi)
        bands_above = sum(1 for i in selected if result.band_ranges[i][0] > engine.e_fermi)
        bands_crossing = len(selected) - bands_below - bands_above

        print(f"✓ Selected bands span Fermi level:")
        print(f"  Bands below E_F: {bands_below}")
        print(f"  Bands crossing E_F: {bands_crossing}")
        print(f"  Bands above E_F: {bands_above}")

    # Check 3: Average distance from Fermi
    avg_distance = np.mean([abs((result.band_ranges[i][0] + result.band_ranges[i][1])/2 - engine.e_fermi)
                            for i in selected])
    if avg_distance > 10.0:
        print(f"⚠ Warning: Average distance from Fermi level is large ({avg_distance:.2f} eV)")
        print(f"  Consider narrowing your energy window.")
    else:
        print(f"✓ Average distance from E_F: {avg_distance:.2f} eV (good)")

    # Select projection orbitals
    print(f"\nStep 9: Selecting projection orbitals...")
    print("-" * 80)

    engine.select_projections(verbose=True)

    # Write data files using .nnkp neighbors
    print(f"\nStep 10: Writing data files using {nnkp_file} neighbors...")
    print("-" * 80)

    # Write only the data files, not .win (already exists)
    engine.write_files(
        verbose=True,
        write_win=False,  # Don't overwrite .win
        use_nnkp=True,    # Use .nnkp neighbors (CRITICAL!)
    )

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.eig")
    print(f"✓ Created: {args.seedname}.amn")
    print(f"✓ Created: {args.seedname}.mmn")
    print()
    print("All files generated with correct neighbor structure from .nnkp!")
    print()
    print("NEXT STEP:")
    print(f"  Run Wannier90 to generate maximally localized Wannier functions:")
    print(f"  → wannier90.x {args.seedname}")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="LCAO-to-Wannier90 Two-Stage Workflow Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  Stage 1: Create .win parameter file
    python %(prog)s --stage 1 --input material.out --seedname material

  Then run Wannier90 preprocessing:
    wannier90.x -pp material

  Stage 2: Create data files using .nnkp neighbors
    python %(prog)s --stage 2 --input material.out --seedname material

  Then run Wannier90:
    wannier90.x material

EXAMPLES:
  # Bismuth with default settings
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth
  wannier90.x -pp bismuth
  python %(prog)s --stage 2 --input Bi.out --seedname bismuth
  wannier90.x bismuth

  # Custom energy window
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --window -6 2
  wannier90.x -pp bismuth
  python %(prog)s --stage 2 --input Bi.out --seedname bismuth --window -6 2
  wannier90.x bismuth
"""
    )

    # Required arguments
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                        help='Stage 1: Create .win | Stage 2: Create .eig/.amn/.mmn')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CRYSTAL/LCAO output file')
    parser.add_argument('--seedname', '-s', type=str, required=True,
                        help='Seedname for Wannier90 files (e.g., "material")')

    # Optional arguments
    parser.add_argument('--window', type=float, nargs=2, metavar=('E_MIN', 'E_MAX'),
                        help='Energy window in eV relative to Fermi level (default: -5.0 3.0)')
    parser.add_argument('--projections', type=str, nargs='+',
                        help='Wannier90 projection strings (default: random)')
    parser.add_argument('--bands-plot', action='store_true',
                        help='Enable band structure plotting in Wannier90')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel computation')

    args = parser.parse_args()

    # Execute appropriate stage
    if args.stage == 1:
        stage1_create_win(args)
    elif args.stage == 2:
        stage2_create_data_files(args)


if __name__ == '__main__':
    main()
