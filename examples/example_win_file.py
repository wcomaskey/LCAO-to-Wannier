#!/usr/bin/env python3
"""
Example: Generating Wannier90 .win Input Files

This example demonstrates how to generate complete Wannier90 input files
including the mandatory .win parameter file.

Features demonstrated:
- Automatic .win file generation from engine
- Parsing atomic positions from CRYSTAL output
- Setting up band structure plotting
- Using predefined high-symmetry paths
- Manual .win file configuration
"""

import sys
import numpy as np
from lcao_wannier import (
    Wannier90Engine,
    write_win_file,
    create_win_config_from_engine,
    parse_atoms_from_crystal_output,
    Wannier90WinConfig,
    KPATH_HEXAGONAL_2D,
    KPATH_SQUARE_2D,
)


def example_simple_win_file():
    """
    Example 1: Simple synthetic system with automatic .win generation.
    """
    print("=" * 70)
    print("Example 1: Simple System with Automatic .win File Generation")
    print("=" * 70)

    # Create simple tight-binding model
    lattice_vectors = np.eye(3) * 5.0  # 5 Angstrom cubic cell

    # Simple Hamiltonian and overlap matrices
    real_space_matrices = {}

    # On-site (R = 0)
    real_space_matrices[(0, 0, 0)] = {
        'H': np.diag([0.0, 1.0, 2.0, 3.0]) + 0j,
        'S': np.eye(4) + 0j
    }

    # Nearest-neighbor hopping
    t = -0.5
    for R in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        real_space_matrices[R] = {
            'H': t * np.eye(4) + 0j,
            'S': 0.1 * np.eye(4) + 0j
        }

    # Initialize engine
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(4, 4, 4),
        lattice_vectors=lattice_vectors,
        num_wann=4,
        seedname='example_simple'
    )

    # Solve eigenvalue problems
    engine.solve_all_kpoints(parallel=False)

    # Write all files including .win (default behavior)
    print("\nWriting Wannier90 files...")
    engine.write_files(
        verbose=True,
        write_win=True,
        spinors=False,  # No SOC
        bands_plot=False
    )

    print("\n✓ Created: example_simple.win, .eig, .amn, .mmn")
    print("\nYou can now run: wannier90.x example_simple")
    print()


def example_with_atoms_and_bands():
    """
    Example 2: System with atomic positions and band structure plotting.
    """
    print("=" * 70)
    print("Example 2: 2D System with Atoms and Band Structure")
    print("=" * 70)

    # 2D hexagonal lattice (e.g., graphene-like)
    a = 2.46  # Angstrom
    lattice_vectors = np.array([
        [a, 0.0, 0.0],
        [a/2, a*np.sqrt(3)/2, 0.0],
        [0.0, 0.0, 10.0]  # Large spacing in z
    ])

    # Create minimal tight-binding model
    real_space_matrices = {}
    num_orbitals = 4

    real_space_matrices[(0, 0, 0)] = {
        'H': np.diag(np.random.rand(num_orbitals)) + 0j,
        'S': np.eye(num_orbitals) + 0j
    }

    # Atomic positions (fractional coordinates)
    atoms = [
        ('C', np.array([0.333, 0.667, 0.5])),
        ('C', np.array([0.667, 0.333, 0.5])),
    ]

    # Initialize engine
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(8, 8, 1),
        lattice_vectors=lattice_vectors,
        num_wann=4,
        seedname='example_graphene'
    )

    # Solve eigenvalue problems
    engine.solve_all_kpoints(parallel=False)

    # Write files with band structure plotting enabled
    print("\nWriting Wannier90 files with band structure setup...")
    engine.write_files(
        verbose=True,
        write_win=True,
        atoms=atoms,
        spinors=False,
        bands_plot=True,
        kpoint_path=KPATH_HEXAGONAL_2D  # Predefined path for hexagonal lattice
    )

    print("\n✓ Created: example_graphene.win, .eig, .amn, .mmn")
    print("✓ Band structure path included: Γ-M-K-Γ")
    print("\nRun: wannier90.x example_graphene")
    print("Then: wannier90.x -pp example_graphene  # For band plotting")
    print()


def example_manual_win_config():
    """
    Example 3: Manual configuration of .win file with custom parameters.
    """
    print("=" * 70)
    print("Example 3: Manual .win File Configuration")
    print("=" * 70)

    # Create k-point grid
    k_grid = (6, 6, 6)
    num_kpts = np.prod(k_grid)

    # Generate k-points (Monkhorst-Pack)
    kpoints = []
    for i in range(k_grid[0]):
        for j in range(k_grid[1]):
            for k in range(k_grid[2]):
                kpt = np.array([
                    (i + 0.5) / k_grid[0],
                    (j + 0.5) / k_grid[1],
                    (k + 0.5) / k_grid[2]
                ])
                kpoints.append(kpt)
    kpoints = np.array(kpoints)

    # Manual configuration
    config = Wannier90WinConfig(
        num_wann=10,
        num_bands=10,  # No disentanglement
        mp_grid=k_grid,
        lattice_vectors=np.eye(3) * 5.0,
        kpoints=kpoints,
        # Optional: atoms
        atoms=[
            ('Si', np.array([0.0, 0.0, 0.0])),
            ('Si', np.array([0.25, 0.25, 0.25])),
        ],
        # Optional: custom projections
        projections=[
            'Si:sp3',
            'Si:sp3',
        ],
        # Control parameters
        num_iter=500,
        conv_tol=1e-10,
        write_hr=True,
        write_xyz=True,
        spinors=False,
        # Output options
        hr_cutoff=5.0,  # Only write H(R) for R < 5 Angstrom
        # Advanced: additional keywords
        additional_keywords={
            'guiding_centres': 'T',
            'num_dump_cycles': 100,
        }
    )

    # Write .win file
    print("\nWriting custom .win file...")
    filepath = write_win_file('example_manual', config, verbose=True)

    print(f"\n✓ Created: {filepath}")
    print("✓ Custom configuration with sp3 projections")
    print("✓ Additional Wannier90 keywords included")
    print()


def example_from_crystal_output():
    """
    Example 4: Parse atoms from CRYSTAL output and generate .win file.
    """
    print("=" * 70)
    print("Example 4: Extracting Atoms from CRYSTAL Output")
    print("=" * 70)

    # This example shows how to parse a real CRYSTAL output file
    crystal_file = "example_lcao_output.txt"  # Placeholder

    print(f"\nNote: This example requires a CRYSTAL output file.")
    print(f"Expected file: {crystal_file}")

    try:
        with open(crystal_file, 'r') as f:
            lines = f.readlines()

        # Parse atoms and lattice from CRYSTAL output
        atoms, lattice_vectors = parse_atoms_from_crystal_output(lines)

        print(f"\n✓ Parsed {len(atoms)} atoms from CRYSTAL output")
        print(f"  Lattice vectors extracted")

        # Continue with engine setup and file generation...
        # (Would require full LCAO matrices from CRYSTAL)

    except FileNotFoundError:
        print(f"\n⚠ File not found: {crystal_file}")
        print("  This is just a demonstration of the API.")
        print("\nUsage:")
        print("  from lcao_wannier import parse_atoms_from_crystal_output")
        print("  with open('crystal.out', 'r') as f:")
        print("      lines = f.readlines()")
        print("  atoms, lattice = parse_atoms_from_crystal_output(lines)")

    print()


def example_spinor_system():
    """
    Example 5: System with spin-orbit coupling (spinor wavefunctions).
    """
    print("=" * 70)
    print("Example 5: Spin-Orbit Coupled System (Spinors)")
    print("=" * 70)

    # For SOC, matrices are typically 2N×2N (spin-block)
    num_orbitals = 8  # 4 spatial orbitals × 2 spin

    lattice_vectors = np.eye(3) * 5.0
    real_space_matrices = {}

    # Simple SOC Hamiltonian
    real_space_matrices[(0, 0, 0)] = {
        'H': np.diag(np.random.rand(num_orbitals)) + 0j,
        'S': np.eye(num_orbitals) + 0j
    }

    # Initialize engine
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(4, 4, 4),
        lattice_vectors=lattice_vectors,
        num_wann=8,
        seedname='example_soc'
    )

    # Solve eigenvalue problems
    engine.solve_all_kpoints(parallel=False)

    # Write files with spinor flag enabled
    print("\nWriting Wannier90 files for SOC system...")
    engine.write_files(
        verbose=True,
        write_win=True,
        spinors=True,  # Critical for SOC!
        bands_plot=False
    )

    print("\n✓ Created: example_soc.win, .eig, .amn, .mmn")
    print("✓ Spinors flag enabled in .win file")
    print("\nNote: Wannier90 will expect spinor wavefunctions")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Wannier90 .win File Examples" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    examples = [
        ("1", "Simple automatic .win generation", example_simple_win_file),
        ("2", "With atoms and band structure", example_with_atoms_and_bands),
        ("3", "Manual .win configuration", example_manual_win_config),
        ("4", "Parse atoms from CRYSTAL", example_from_crystal_output),
        ("5", "Spin-orbit coupled system", example_spinor_system),
    ]

    if len(sys.argv) > 1:
        # Run specific example
        choice = sys.argv[1]
        for num, desc, func in examples:
            if num == choice:
                func()
                return
        print(f"Invalid choice: {choice}")
        print("Available examples: 1-5")
    else:
        # Run all examples
        for num, desc, func in examples:
            func()
            input("Press Enter to continue to next example...")
            print("\n")


if __name__ == "__main__":
    main()
