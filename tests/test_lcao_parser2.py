"""
CORRECT VERSION - Test Script: LCAO Parser to Wannier90 Integration

Created from scratch following the actual parser logic.

Key structure (from parser analysis):
1. Overlap matrices (NO spin labels)
2. ALPHA_ALPHA label → all ALPHA_ALPHA Fock matrices (all lattice vectors)
3. BETA_BETA label → all BETA_BETA Fock matrices (all lattice vectors)
4. ALPHA_BETA label → all ALPHA_BETA Fock matrices (all lattice vectors)
5. BETA_ALPHA label → all BETA_ALPHA Fock matrices (all lattice vectors)

Each spin label appears ONCE and applies to all subsequent Fock matrices
until the next spin label.

Usage:
    python test_lcao_parser.py
    python test_lcao_parser.py <lcao_file>
    python test_lcao_parser.py <lcao_file> --k-grid 6 6 6
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import (
    Wannier90Engine,
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
)


# ============================================================================
# Example LCAO Data Generator - CORRECT STRUCTURE
# ============================================================================

def create_example_lcao_output():
    """
    Create example LCAO output with PHYSICALLY REALISTIC matrices.
    
    Key properties ensured:
    1. Overlap matrices: Real, symmetric, positive definite
    2. Fock matrices: Hermitian (lower triangle only for real part)
    3. Off-diagonal spin blocks: Zero (no spin-orbit coupling)
    4. Proper relationship: H(-R) = H(R)^† for Hermiticity in k-space
    
    Structure follows parser requirements:
    1. Lattice vectors
    2. All overlap matrices (no spin labels)
    3. ALPHA_ALPHA label → all ALPHA_ALPHA Fock matrices
    4. BETA_BETA label → all BETA_BETA Fock matrices
    5. ALPHA_BETA label → all ALPHA_BETA Fock matrices (zeros)
    6. BETA_ALPHA label → all BETA_ALPHA Fock matrices (zeros)
    
    Returns
    -------
    str
        Example LCAO output as a string
    """
    example_output = """
 DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)
         5.0000000    0.0000000    0.0000000
         0.0000000    5.0000000    0.0000000
         0.0000000    0.0000000    5.0000000

 OVERLAP MATRIX - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     1.0000000E+00
   2     0.0000000E+00  1.0000000E+00
   3     0.0000000E+00  0.0000000E+00  1.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  1.0000000E+00

 OVERLAP MATRIX - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     1.0000000E-01
   2     0.0000000E+00  1.0000000E-01
   3     0.0000000E+00  0.0000000E+00  1.0000000E-01
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  1.0000000E-01

 ALPHA_ALPHA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1    -5.0000000E+00
   2     0.0000000E+00 -4.0000000E+00
   3     0.0000000E+00  0.0000000E+00 -3.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00 -2.0000000E+00

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1    -5.0000000E-01
   2     0.0000000E+00 -4.0000000E-01
   3     0.0000000E+00  0.0000000E+00 -3.0000000E-01
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00 -2.0000000E-01

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 BETA_BETA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1    -5.0000000E+00
   2     0.0000000E+00 -4.0000000E+00
   3     0.0000000E+00  0.0000000E+00 -3.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00 -2.0000000E+00

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1    -5.0000000E-01
   2     0.0000000E+00 -4.0000000E-01
   3     0.0000000E+00  0.0000000E+00 -3.0000000E-01
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00 -2.0000000E-01

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 ALPHA_BETA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 BETA_ALPHA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00
"""
    return example_output


# ============================================================================
# LCAO File Parser
# ============================================================================

def parse_lcao_file(filepath):
    """
    Parse LCAO output file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to LCAO output file
        
    Returns
    -------
    matrices : list
        List of parsed matrix dictionaries
    lattice_vectors : ndarray
        Lattice vectors (3x3 array)
    """
    print(f"\nParsing LCAO file: {filepath}")
    print("=" * 70)
    
    # Read file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"  Read {len(lines)} lines from file")
    
    # Parse matrices and lattice vectors
    matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
    
    # Convert lattice_vectors from list to numpy array
    if lattice_vectors_list is None:
        raise ValueError("Could not parse lattice vectors from file")
    
    lattice_vectors = np.array(lattice_vectors_list)
    
    print(f"  Found {len(matrices)} matrix blocks")
    
    # DEBUG: Show what was parsed
    print("\n  DEBUG: Parsed matrices summary:")
    overlap_count = sum(1 for m in matrices if m['type'] == 'overlap')
    fock_count = sum(1 for m in matrices if m['type'] == 'fock')
    none_count = sum(1 for m in matrices if m['data'] is None)
    print(f"    Overlap matrices: {overlap_count}")
    print(f"    Fock matrices: {fock_count}")
    print(f"    Matrices with None data: {none_count}")
    
    if fock_count > 0:
        print(f"\n  DEBUG: Fock matrices by spin channel:")
        for spin in ['ALPHA_ALPHA', 'BETA_BETA', 'ALPHA_BETA', 'BETA_ALPHA']:
            count = sum(1 for m in matrices if m['type'] == 'fock' and m.get('spin_channel') == spin and m['data'] is not None)
            print(f"    {spin}: {count} matrices")
    
    print(f"  Lattice vectors shape: {lattice_vectors.shape}")
    print(f"\n  Lattice vectors (Angstrom):")
    for i, vec in enumerate(lattice_vectors):
        print(f"    a{i+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")
    
    return matrices, lattice_vectors


def prepare_real_space_data(matrices, lattice_vectors):
    """
    Prepare real-space matrices for Wannier90Engine.
    
    Parameters
    ----------
    matrices : list
        Parsed matrices from LCAO file
    lattice_vectors : ndarray
        Lattice vectors (3x3)
        
    Returns
    -------
    real_space_matrices : dict
        Dictionary with structure: {(i,j,k): {'H': H_matrix, 'S': S_matrix}}
    N_basis : int
        Number of basis functions (orbitals) per spin
    """
    print("\nPreparing real-space matrices:")
    print("=" * 70)
    
    # Filter out matrices with None data and provide diagnostics
    valid_matrices = []
    invalid_count = 0
    
    for mat in matrices:
        if mat['data'] is None:
            invalid_count += 1
            print(f"  Warning: Skipping matrix with None data - "
                  f"Type: {mat['type']}, R={mat['lattice_vector']}, "
                  f"Spin: {mat.get('spin_channel', 'N/A')}")
        else:
            valid_matrices.append(mat)
    
    if invalid_count > 0:
        print(f"\n  Skipped {invalid_count} matrices with invalid (None) data")
        print(f"  Proceeding with {len(valid_matrices)} valid matrices")
    
    if len(valid_matrices) == 0:
        raise ValueError(
            "No valid matrices found! The parser could not extract any matrix data.\n"
            "This usually means:\n"
            "  1. The file format doesn't match what the parser expects\n"
            "  2. The matrix data sections are malformed\n"
            "  3. The example data generation has issues\n"
            "Please check the file format and parser configuration."
        )
    
    # Organize matrices by lattice vector and type
    H_dict = {}  # Hamiltonian (Fock) matrices
    S_dict = {}  # Overlap matrices
    
    N_basis = None
    
    print("\n  DEBUG: Organizing matrices by type and lattice vector...")
    for idx, mat in enumerate(valid_matrices):
        matrix_type = mat['type']
        lattice_vec = tuple(mat['lattice_vector'])
        data = mat['data']
        spin_channel = mat.get('spin_channel', 'N/A')
        
        print(f"    Matrix {idx}: type={matrix_type}, R={lattice_vec}, spin={spin_channel}, shape={data.shape}")
        
        # Determine basis size from first valid matrix
        if N_basis is None:
            N_basis = data.shape[0]
            print(f"\n  Basis size (orbitals per spin): {N_basis}")
        
        if matrix_type == 'fock':
            spin_channel = mat['spin_channel']
            if lattice_vec not in H_dict:
                H_dict[lattice_vec] = {}
            H_dict[lattice_vec][spin_channel] = data
            
        elif matrix_type == 'overlap':
            S_dict[lattice_vec] = data
    
    print(f"  Found {len(H_dict)} unique lattice vectors for Hamiltonian")
    print(f"  Found {len(S_dict)} unique lattice vectors for Overlap")
    
    # Check which spin channels we have
    if len(H_dict) == 0:
        raise ValueError("No Hamiltonian matrices found after parsing!")
    
    example_R = list(H_dict.keys())[0]
    spin_channels = list(H_dict[example_R].keys())
    print(f"  Spin channels present: {spin_channels}")
    
    # Create spin-block matrices
    print("\n  Creating 2N×2N spin-block matrices...")
    result = create_spin_block_matrices(H_dict, S_dict, N_basis, lattice_vectors)
    
    # DEBUG: Check what we got back
    print(f"\n  DEBUG: Type of result from create_spin_block_matrices: {type(result)}")
    if isinstance(result, tuple):
        print(f"  DEBUG: Tuple has {len(result)} elements")
        print(f"  DEBUG: Element 0 type: {type(result[0])}, length: {len(result[0])}")
        print(f"  DEBUG: Element 1 type: {type(result[1])}, length: {len(result[1])}")
        if len(result[0]) > 0:
            print(f"  DEBUG: First H element: type={type(result[0][0])}, len={len(result[0][0]) if isinstance(result[0][0], tuple) else 'N/A'}")
            if isinstance(result[0][0], tuple) and len(result[0][0]) == 2:
                print(f"  DEBUG: First H tuple: R_cart shape={result[0][0][0].shape if hasattr(result[0][0][0], 'shape') else 'N/A'}, matrix shape={result[0][0][1].shape}")
    
    # Handle the actual return format: (H_full_list, S_full_list)
    # where H_full_list = [(R_cartesian, H_matrix), ...]
    # and S_full_list = [(R_cartesian, S_matrix), ...]
    if isinstance(result, tuple) and len(result) == 2:
        H_full_list, S_full_list = result
        
        # Convert to dictionary format: {R_integer: {'H': H_matrix, 'S': S_matrix}}
        real_space_matrices = {}
        
        # Process H matrices
        print(f"\n  Converting {len(H_full_list)} Hamiltonian matrices to dictionary...")
        for R_cartesian, H_matrix in H_full_list:
            # Convert Cartesian to integer lattice coordinates
            R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
            R_tuple = tuple(R_integer)
            
            if R_tuple not in real_space_matrices:
                real_space_matrices[R_tuple] = {}
            real_space_matrices[R_tuple]['H'] = H_matrix
            print(f"    H: R_cart={R_cartesian} -> R_int={R_tuple}, shape={H_matrix.shape}")
        
        # Process S matrices
        print(f"\n  Converting {len(S_full_list)} Overlap matrices to dictionary...")
        for R_cartesian, S_matrix in S_full_list:
            # Convert Cartesian to integer lattice coordinates
            R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
            R_tuple = tuple(R_integer)
            
            if R_tuple not in real_space_matrices:
                real_space_matrices[R_tuple] = {}
            real_space_matrices[R_tuple]['S'] = S_matrix
            print(f"    S: R_cart={R_cartesian} -> R_int={R_tuple}, shape={S_matrix.shape}")
        
        print(f"\n  Created {len(real_space_matrices)} real-space matrix pairs")
        print(f"  Matrix dimensions: {list(real_space_matrices.values())[0]['H'].shape}")
    else:
        raise ValueError(f"Unexpected return type from create_spin_block_matrices: {type(result)}")
    
    # Verify Hermiticity
    print("\n  Verifying matrix properties:")
    max_H_error = 0.0
    max_S_error = 0.0
    
    for R, mats in real_space_matrices.items():
        H = mats['H']
        S = mats['S']
        
        H_error = np.max(np.abs(H - H.conj().T))
        S_error = np.max(np.abs(S - S.conj().T))
        
        max_H_error = max(max_H_error, H_error)
        max_S_error = max(max_S_error, S_error)
    
    print(f"    Max H Hermiticity error: {max_H_error:.2e}")
    print(f"    Max S Hermiticity error: {max_S_error:.2e}")
    
    if max_H_error > 1e-10:
        print(f"    WARNING: H matrices have significant Hermiticity errors!")
    if max_S_error > 1e-10:
        print(f"    WARNING: S matrices have significant Hermiticity errors!")
    
    return real_space_matrices, N_basis


# ============================================================================
# Complete Test Workflow
# ============================================================================

def test_lcao_to_wannier90(lcao_file='example_lcao_output.txt', 
                           k_grid=(4, 4, 4),
                           num_wann=None,
                           seedname='my_material',
                           parallel=True):
    """
    Complete test of LCAO to Wannier90 workflow.
    
    Parameters
    ----------
    lcao_file : str
        Path to LCAO output file, or '--example' to use synthetic data
    k_grid : tuple of int
        K-point grid dimensions (e.g., (4, 4, 4))
    num_wann : int or None
        Number of Wannier functions (defaults to all orbitals)
    seedname : str
        Output file prefix
    parallel : bool
        Use parallel processing for k-point loops
    
    Returns
    -------
    engine : Wannier90Engine
        The engine object with results
    results : dict
        Results dictionary from engine.run()
    """
    print("\n" + "=" * 70)
    print("LCAO TO WANNIER90 - COMPLETE WORKFLOW TEST")
    print("=" * 70)
    
    # Handle example data
    if lcao_file == 'example_lcao_output.txt' and not os.path.exists(lcao_file):
        print("Using example LCAO data (synthetic)")
        example_data = create_example_lcao_output()
        
        # Write to temporary file
        temp_file = 'example_lcao_output.txt'
        with open(temp_file, 'w') as f:
            f.write(example_data)
        lcao_file = temp_file
    else:
        # Check if file exists
        if not os.path.exists(lcao_file):
            raise FileNotFoundError(f"LCAO file not found: {lcao_file}")
    
    # Parse LCAO file
    matrices, lattice_vectors = parse_lcao_file(lcao_file)
    
    # Prepare real-space data
    real_space_matrices, N_basis = prepare_real_space_data(matrices, lattice_vectors)
    
    # Set num_wann if not provided
    if num_wann is None:
        # Use 2*N_basis for spin-polarized (full 2N×2N matrix)
        num_wann = 2 * N_basis
        print(f"\nUsing all {num_wann} orbitals as Wannier functions")
    else:
        print(f"\nUsing {num_wann} Wannier functions")
    
    # Initialize Wannier90 engine
    print("\nInitializing Wannier90 engine:")
    print("=" * 70)
    print(f"  K-grid: {k_grid[0]} × {k_grid[1]} × {k_grid[2]}")
    print(f"  Number of Wannier functions: {num_wann}")
    print(f"  Output seedname: {seedname}")
    print(f"  Parallel processing: {parallel}")
    
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=k_grid,
        lattice_vectors=lattice_vectors,
        num_wann=num_wann,
        seedname=seedname
    )
    
    # Run the workflow
    print("\nRunning Wannier90 workflow:")
    print("=" * 70)
    
    results = engine.run(parallel=parallel, verify=True)
    
    # Print summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  {seedname}.eig  - Eigenvalues (band energies)")
    print(f"  {seedname}.amn  - Projection matrices")
    print(f"  {seedname}.mmn  - Overlap matrices (k-neighbors)")
    print(f"\nThese files can now be used with Wannier90 for:")
    print(f"  - Wannier function localization")
    print(f"  - Band structure interpolation")
    print(f"  - Berry phase calculations")
    print(f"  - Transport properties")
    print("=" * 70)
    
    return engine, results


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    """Command-line interface for test script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test LCAO to Wannier90 workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with example data (default)
  python test_lcao_parser.py
  
  # Run with your LCAO file
  python test_lcao_parser.py your_file.txt
  
  # Customize parameters
  python test_lcao_parser.py your_file.txt --k-grid 6 6 6 --num-wann 10
        '''
    )
    
    parser.add_argument('lcao_file', nargs='?', default='example_lcao_output.txt',
                        help='Path to LCAO output file (default: generate example)')
    parser.add_argument('--k-grid', type=int, nargs=3, default=[4, 4, 4],
                        metavar=('NX', 'NY', 'NZ'),
                        help='K-point grid dimensions (default: 4 4 4)')
    parser.add_argument('--num-wann', type=int, default=None,
                        help='Number of Wannier functions (default: all orbitals)')
    parser.add_argument('--seedname', type=str, default='my_material',
                        help='Output file prefix (default: my_material)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Handle no arguments case
    if len(sys.argv) == 1:
        print("No arguments provided. Running with example data...")
        print("Use --help for usage information.")
    
    try:
        engine, results = test_lcao_to_wannier90(
            lcao_file=args.lcao_file,
            k_grid=tuple(args.k_grid),
            num_wann=args.num_wann,
            seedname=args.seedname,
            parallel=not args.no_parallel
        )
        
        print("\n✓ Test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
