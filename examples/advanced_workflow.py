"""
Advanced Workflow Example for LCAO-Wannier Package

This example demonstrates the complete workflow from parsing CRYSTAL
output files to generating Wannier90 inputs.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import (
    Wannier90Engine,
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
    prepare_real_space_matrices
)
from lcao_wannier.utils import (
    organize_matrices_by_lattice_vector,
    print_matrix_summary,
    print_calculation_info
)


def parse_crystal_output(filename):
    """
    Parse CRYSTAL output file and prepare data for Wannier90 engine.
    
    Parameters
    ----------
    filename : str
        Path to CRYSTAL output file
    
    Returns
    -------
    real_space_matrices : dict
        Formatted matrices ready for Wannier90Engine
    lattice_vectors : ndarray
        Real-space lattice vectors
    """
    print(f"Parsing CRYSTAL output file: {filename}")
    
    # Read file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse matrices and lattice vectors
    matrices, direct_lattice_vectors = parse_overlap_and_fock_matrices(lines)
    
    if direct_lattice_vectors is None:
        raise ValueError("Could not find direct lattice vectors in the file")
    
    lattice_vectors = np.array(direct_lattice_vectors)
    print(f"  ✓ Found {len(matrices)} matrices")
    print(f"  ✓ Lattice vectors: {lattice_vectors.shape}")
    
    # Organize matrices by lattice vector and spin channel
    H_R_dict, S_R_dict = organize_matrices_by_lattice_vector(matrices)
    
    print(f"  ✓ Organized into {len(H_R_dict)} Hamiltonian R-vectors")
    print(f"  ✓ Organized into {len(S_R_dict)} overlap R-vectors")
    
    # Determine basis size
    first_S = next(iter(S_R_dict.values()))
    N_basis = first_S.shape[0]
    print(f"  ✓ Basis size: {N_basis}")
    
    # Create spin-block matrices
    print("Creating spin-block matrices...")
    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict, S_R_dict, N_basis, direct_lattice_vectors, PRINTOUT=False
    )
    
    print(f"  ✓ Created {len(H_full_list)} Hamiltonian matrices")
    print(f"  ✓ Created {len(S_full_list)} overlap matrices")
    print(f"  ✓ Full matrix size: {H_full_list[0][1].shape}")
    
    # Prepare for Wannier90 engine
    print("Preparing data for Wannier90 engine...")
    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors
    )
    
    print(f"  ✓ Prepared {len(real_space_matrices)} R-vectors")
    
    return real_space_matrices, lattice_vectors


def run_complete_workflow(crystal_file, k_grid, num_wann, seedname='wannier90'):
    """
    Run the complete workflow from CRYSTAL output to Wannier90 files.
    
    Parameters
    ----------
    crystal_file : str
        Path to CRYSTAL output file
    k_grid : tuple of 3 ints
        K-point grid dimensions
    num_wann : int
        Number of Wannier functions
    seedname : str
        Output file prefix
    """
    print("\n" + "=" * 70)
    print("ADVANCED WORKFLOW - CRYSTAL TO WANNIER90")
    print("=" * 70)
    
    # Step 1: Parse CRYSTAL output
    print("\n" + "-" * 70)
    print("STEP 1: Parsing CRYSTAL Output")
    print("-" * 70)
    
    real_space_matrices, lattice_vectors = parse_crystal_output(crystal_file)
    
    # Print matrix summary
    print_matrix_summary(real_space_matrices)
    
    # Step 2: Initialize and run Wannier90 engine
    print("\n" + "-" * 70)
    print("STEP 2: Running Wannier90 Engine")
    print("-" * 70)
    
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=k_grid,
        lattice_vectors=lattice_vectors,
        num_wann=num_wann,
        seedname=seedname
    )
    
    # Run workflow with verification
    results = engine.run(parallel=True, verify=True)
    
    # Step 3: Analyze results
    print("\n" + "-" * 70)
    print("STEP 3: Analysis")
    print("-" * 70)
    
    # Get band structure
    kpoints, eigenvalues = engine.get_band_structure()
    print(f"Band structure shape: {eigenvalues.shape}")
    print(f"Energy range: [{np.min(eigenvalues):.4f}, {np.max(eigenvalues):.4f}]")
    
    # Get density of states
    energies, dos = engine.get_density_of_states(num_bins=100)
    print(f"DOS computed with {len(energies)} energy bins")
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced workflow for CRYSTAL to Wannier90 conversion'
    )
    parser.add_argument(
        'crystal_file',
        help='Path to CRYSTAL output file'
    )
    parser.add_argument(
        '--kgrid',
        type=int,
        nargs=3,
        default=[6, 6, 6],
        help='K-point grid dimensions (default: 6 6 6)'
    )
    parser.add_argument(
        '--num-wann',
        type=int,
        default=10,
        help='Number of Wannier functions (default: 10)'
    )
    parser.add_argument(
        '--seedname',
        type=str,
        default='wannier90',
        help='Output file prefix (default: wannier90)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.crystal_file):
        print(f"Error: File '{args.crystal_file}' not found!")
        sys.exit(1)
    
    # Run workflow
    try:
        run_complete_workflow(
            crystal_file=args.crystal_file,
            k_grid=tuple(args.kgrid),
            num_wann=args.num_wann,
            seedname=args.seedname
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If no arguments provided, print usage
    if len(sys.argv) == 1:
        print("=" * 70)
        print("ADVANCED WORKFLOW EXAMPLE")
        print("=" * 70)
        print("\nUsage:")
        print("  python advanced_workflow.py <crystal_output_file> [options]")
        print("\nOptions:")
        print("  --kgrid NX NY NZ      K-point grid (default: 6 6 6)")
        print("  --num-wann N          Number of Wannier functions (default: 10)")
        print("  --seedname NAME       Output file prefix (default: wannier90)")
        print("\nExample:")
        print("  python advanced_workflow.py my_calc.out --kgrid 8 8 8 --num-wann 20")
        print("=" * 70)
        sys.exit(0)
    
    main()
