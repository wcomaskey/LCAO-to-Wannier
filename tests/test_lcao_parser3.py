"""
Test Script: LCAO Parser to Wannier90 Integration

UPDATED VERSION - Compatible with new parser.py that uses:
- Global Pair Symmetry Construction (R/-R pairing)
- Returns (H_full_list, S_full_list) tuples from create_spin_block_matrices

This script tests the complete workflow:
1. Parsing LCAO output files
2. Organizing matrices by lattice vector  
3. Creating 2Nx2N spin-block matrices with R/-R pairing
4. Converting to dictionary format for Wannier90 engine
5. Running verification checks

Usage:
    python test_lcao_parser.py <lcao_output_file>
    python test_lcao_parser.py --example
    python test_lcao_parser.py --help
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from lcao_wannier package
try:
    from lcao_wannier.parser import (
        parse_overlap_and_fock_matrices,
        create_spin_block_matrices,
        fill_raw_matrix,
    )
    from lcao_wannier.utils import (
        prepare_real_space_matrices,
        organize_matrices_by_lattice_vector,
        get_basis_size,
        verify_matrix_symmetry,
        check_matrix_consistency,
        print_matrix_summary,
    )
    from lcao_wannier.verification import (
        verify_real_space_symmetry,
        verify_hermiticity,
        verify_orthonormality,
        run_all_verifications,
    )
    from lcao_wannier import Wannier90Engine
    FULL_PACKAGE = True
except ImportError as e:
    print(f"Warning: Could not import full lcao_wannier package: {e}")
    print("Attempting to import modules directly...")
    FULL_PACKAGE = False
    
    # Try direct imports for standalone testing
    try:
        from parser import (
            parse_overlap_and_fock_matrices,
            create_spin_block_matrices,
        )
        from utils import (
            prepare_real_space_matrices,
            organize_matrices_by_lattice_vector,
            get_basis_size,
            verify_matrix_symmetry,
        )
        from verification import (
            verify_real_space_symmetry,
            verify_hermiticity,
        )
    except ImportError as e2:
        print(f"Error: Could not import required modules: {e2}")
        print("Make sure parser.py, utils.py, and verification.py are in the path.")
        sys.exit(1)


# ============================================================================
# Example LCAO Data Generator
# ============================================================================

def create_example_lcao_output():
    """
    Create example LCAO output with physically realistic matrices.
    
    Key properties ensured for Global Pair Symmetry Construction:
    1. Overlap matrices: Real, symmetric, positive definite
    2. Fock matrices: Lower triangular only (parser expects this)
    3. Off-diagonal spin blocks: Zero (no spin-orbit coupling in this example)
    4. Both R and -R vectors present for proper pairing
    
    Structure follows parser requirements:
    1. Lattice vectors
    2. All overlap matrices (no spin labels)
    3. ALPHA_ALPHA label → all ALPHA_ALPHA Fock matrices
    4. BETA_BETA label → all BETA_BETA Fock matrices
    5. ALPHA_BETA label → all ALPHA_BETA Fock matrices (zeros)
    6. BETA_ALPHA label → all BETA_ALPHA Fock matrices (zeros)
    
    Each spin label appears ONCE and applies to all subsequent Fock matrices.
    """
    
    # Define matrices for R=(0,0,0) - must be Hermitian
    # Using lower triangular format as expected by parser
    
    # Overlap S(0,0,0) - symmetric, positive definite
    S_000 = """   1     1.0000000E+00
   2     1.0000000E-01  1.0000000E+00
   3     5.0000000E-02  1.0000000E-01  1.0000000E+00
   4     2.0000000E-02  5.0000000E-02  1.0000000E-01  1.0000000E+00"""

    # Fock real part H(0,0,0) - lower triangular
    H_real_000 = """   1    -5.0000000E+00
   2     5.0000000E-01 -3.0000000E+00
   3     2.0000000E-01  5.0000000E-01 -4.0000000E+00
   4     1.0000000E-01  2.0000000E-01  5.0000000E-01 -3.5000000E+00"""

    # Fock imag part H(0,0,0) - antisymmetric for Hermitian H
    # Lower triangular values; upper will be -conj(lower)
    H_imag_000 = """   1     0.0000000E+00
   2    -1.0000000E-01  0.0000000E+00
   3    -5.0000000E-02 -1.0000000E-01  0.0000000E+00
   4    -2.0000000E-02 -5.0000000E-02 -1.0000000E-01  0.0000000E+00"""

    # Matrices for R=(1,0,0) - no Hermiticity requirement
    # But H(-R) = H(R)† must hold, so we need R=(-1,0,0) too
    
    S_100 = """   1     5.0000000E-02
   2     2.0000000E-02  5.0000000E-02
   3     1.0000000E-02  2.0000000E-02  5.0000000E-02
   4     5.0000000E-03  1.0000000E-02  2.0000000E-02  5.0000000E-02"""

    H_real_100 = """   1    -2.0000000E-01
   2     5.0000000E-02 -1.5000000E-01
   3     2.0000000E-02  5.0000000E-02 -1.8000000E-01
   4     1.0000000E-02  2.0000000E-02  5.0000000E-02 -1.6000000E-01"""

    H_imag_100 = """   1     0.0000000E+00
   2    -2.0000000E-02  0.0000000E+00
   3    -1.0000000E-02 -2.0000000E-02  0.0000000E+00
   4    -5.0000000E-03 -1.0000000E-02 -2.0000000E-02  0.0000000E+00"""

    # Matrices for R=(-1,0,0) - conjugate transpose of R=(1,0,0)
    # The parser will combine these using Global Pair Symmetry Construction
    
    S_m100 = """   1     5.0000000E-02
   2     2.0000000E-02  5.0000000E-02
   3     1.0000000E-02  2.0000000E-02  5.0000000E-02
   4     5.0000000E-03  1.0000000E-02  2.0000000E-02  5.0000000E-02"""

    # For H(-R) lower triangular to produce H(R)†:
    # H(-R)_lower should be conj(H(R)_upper) = conj(H(R)_lower^T)
    # Real part: same as transpose of H_real_100 lower triangle
    H_real_m100 = """   1    -2.0000000E-01
   2     5.0000000E-02 -1.5000000E-01
   3     2.0000000E-02  5.0000000E-02 -1.8000000E-01
   4     1.0000000E-02  2.0000000E-02  5.0000000E-02 -1.6000000E-01"""

    # Imaginary part: negative of H_imag_100 for conjugate
    H_imag_m100 = """   1     0.0000000E+00
   2     2.0000000E-02  0.0000000E+00
   3     1.0000000E-02  2.0000000E-02  0.0000000E+00
   4     5.0000000E-03  1.0000000E-02  2.0000000E-02  0.0000000E+00"""

    # Zero matrices for off-diagonal spin blocks
    zeros = """   1     0.0000000E+00
   2     0.0000000E+00  0.0000000E+00
   3     0.0000000E+00  0.0000000E+00  0.0000000E+00
   4     0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00"""

    col_header = "                 1              2              3              4\n"

    example_output = f""" DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)
         5.0000000    0.0000000    0.0000000
         0.0000000    5.0000000    0.0000000
         0.0000000    0.0000000    5.0000000

 OVERLAP MATRIX - CELL N.   1(  0  0  0)

{col_header}
{S_000}

 OVERLAP MATRIX - CELL N.   2(  1  0  0)

{col_header}
{S_100}

 OVERLAP MATRIX - CELL N.   3( -1  0  0)

{col_header}
{S_m100}

 ALPHA_ALPHA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

{col_header}
{H_real_000}

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

{col_header}
{H_imag_000}

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

{col_header}
{H_real_100}

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

{col_header}
{H_imag_100}

 FOCK MATRIX (REAL PART) - CELL N.   3( -1  0  0)

{col_header}
{H_real_m100}

 FOCK MATRIX (IMAG PART) - CELL N.   3( -1  0  0)

{col_header}
{H_imag_m100}

 BETA_BETA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

{col_header}
{H_real_000}

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

{col_header}
{H_imag_000}

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

{col_header}
{H_real_100}

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

{col_header}
{H_imag_100}

 FOCK MATRIX (REAL PART) - CELL N.   3( -1  0  0)

{col_header}
{H_real_m100}

 FOCK MATRIX (IMAG PART) - CELL N.   3( -1  0  0)

{col_header}
{H_imag_m100}

 ALPHA_BETA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

{col_header}
{zeros}

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

{col_header}
{zeros}

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

{col_header}
{zeros}

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

{col_header}
{zeros}

 FOCK MATRIX (REAL PART) - CELL N.   3( -1  0  0)

{col_header}
{zeros}

 FOCK MATRIX (IMAG PART) - CELL N.   3( -1  0  0)

{col_header}
{zeros}

 BETA_ALPHA ELECTRONS

 FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)

{col_header}
{zeros}

 FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)

{col_header}
{zeros}

 FOCK MATRIX (REAL PART) - CELL N.   2(  1  0  0)

{col_header}
{zeros}

 FOCK MATRIX (IMAG PART) - CELL N.   2(  1  0  0)

{col_header}
{zeros}

 FOCK MATRIX (REAL PART) - CELL N.   3( -1  0  0)

{col_header}
{zeros}

 FOCK MATRIX (IMAG PART) - CELL N.   3( -1  0  0)

{col_header}
{zeros}
"""
    return example_output


# ============================================================================
# File Parsing
# ============================================================================

def parse_lcao_file(lcao_file):
    """
    Parse an LCAO output file and return matrices and lattice vectors.
    
    Parameters
    ----------
    lcao_file : str
        Path to LCAO output file, or '--example' for synthetic data
    
    Returns
    -------
    matrices : list
        List of parsed matrix dictionaries
    lattice_vectors : ndarray
        3x3 array of lattice vectors (rows are vectors)
    """
    if lcao_file == '--example':
        print("Using example LCAO data (synthetic)")
        content = create_example_lcao_output()
        lines = content.split('\n')
        
        # Save to temp file for debugging
        with open('example_lcao_output.txt', 'w') as f:
            f.write(content)
        print(f"Parsing LCAO file: example_lcao_output.txt")
    else:
        print(f"Parsing LCAO file: {lcao_file}")
        with open(lcao_file, 'r') as f:
            lines = f.readlines()
    
    print("=" * 70)
    print(f"  Read {len(lines)} lines from file")
    
    # Parse matrices using the parser module
    matrices, direct_lattice_vectors = parse_overlap_and_fock_matrices(lines)
    
    print(f"  Found {len(matrices)} matrix blocks")
    
    if direct_lattice_vectors is None:
        raise ValueError("Could not find direct lattice vectors in the file")
    
    # Convert to numpy array
    lattice_vectors = np.array(direct_lattice_vectors)
    
    print(f"  Lattice vectors shape: {lattice_vectors.shape}")
    print(f"  Lattice vectors (Angstrom):")
    print(f"    a1 = [{lattice_vectors[0, 0]:8.4f}, {lattice_vectors[0, 1]:8.4f}, {lattice_vectors[0, 2]:8.4f}]")
    print(f"    a2 = [{lattice_vectors[1, 0]:8.4f}, {lattice_vectors[1, 1]:8.4f}, {lattice_vectors[1, 2]:8.4f}]")
    print(f"    a3 = [{lattice_vectors[2, 0]:8.4f}, {lattice_vectors[2, 1]:8.4f}, {lattice_vectors[2, 2]:8.4f}]")
    
    return matrices, lattice_vectors


# ============================================================================
# Matrix Preparation (Bridge to Engine)
# ============================================================================

def prepare_real_space_data(matrices, lattice_vectors, verbose=True):
    """
    Prepare real-space matrices for Wannier90 engine.
    
    This function:
    1. Organizes raw matrices by lattice vector and spin channel
    2. Creates 2Nx2N spin-block matrices using Global Pair Symmetry Construction
    3. Converts to dictionary format expected by Wannier90Engine
    
    Parameters
    ----------
    matrices : list
        Parsed matrix list from parse_overlap_and_fock_matrices
    lattice_vectors : ndarray
        3x3 lattice vectors array
    verbose : bool
        Print debug information
    
    Returns
    -------
    real_space_matrices : dict
        Dictionary mapping (R_int) -> {'H': H_matrix, 'S': S_matrix}
    N_basis : int
        Number of basis functions (spatial orbitals)
    """
    print("Preparing real-space matrices:")
    print("=" * 70)
    
    # Step 1: Organize matrices by lattice vector
    H_R_dict, S_R_dict = organize_matrices_by_lattice_vector(matrices)
    
    # Determine basis size from overlap matrix
    if S_R_dict:
        first_S = next(iter(S_R_dict.values()))
        N_basis = first_S.shape[0]
    else:
        raise ValueError("No overlap matrices found!")
    
    print(f"  Basis size (orbitals per spin): {N_basis}")
    print(f"  Found {len(H_R_dict)} unique lattice vectors for Hamiltonian")
    print(f"  Found {len(S_R_dict)} unique lattice vectors for Overlap")
    
    # Debug: Show what spin channels are present
    if H_R_dict:
        example_R = list(H_R_dict.keys())[0]
        spin_channels = list(H_R_dict[example_R].keys())
        print(f"  Spin channels present: {spin_channels}")
    
    # Debug: Show all R vectors found
    print(f"  R-vectors for H: {sorted(H_R_dict.keys())}")
    print(f"  R-vectors for S: {sorted(S_R_dict.keys())}")
    
    # Step 2: Create spin-block matrices using Global Pair Symmetry Construction
    print("\n  Creating 2N×2N spin-block matrices...")
    print("  Using Global Pair Symmetry Construction (R/-R pairing)")
    
    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict, 
        S_R_dict, 
        N_basis, 
        lattice_vectors.tolist(),  # Convert to list as expected by function
        PRINTOUT=verbose
    )
    
    if verbose:
        print(f"\n  create_spin_block_matrices returned:")
        print(f"    H_full_list: {len(H_full_list)} entries")
        print(f"    S_full_list: {len(S_full_list)} entries")
    
    # Step 3: Convert to dictionary format for Wannier90Engine
    # The function returns lists of (R_cartesian, matrix) tuples
    # We need to convert to {(R_integer): {'H': H, 'S': S}} format
    
    real_space_matrices = prepare_real_space_matrices(
        H_full_list, 
        S_full_list, 
        lattice_vectors
    )
    
    print(f"\n  Created {len(real_space_matrices)} real-space matrix pairs")
    print(f"  R-vectors: {sorted(real_space_matrices.keys())}")
    
    # Get matrix dimensions
    first_key = next(iter(real_space_matrices))
    if 'H' in real_space_matrices[first_key]:
        dim = real_space_matrices[first_key]['H'].shape[0]
        print(f"  Matrix dimensions: {dim} × {dim} (2N × 2N)")
    
    return real_space_matrices, N_basis


# ============================================================================
# Verification Functions
# ============================================================================

def verify_matrices(real_space_matrices, tolerance=1e-10, verbose=True):
    """
    Verify that constructed matrices satisfy physical requirements.
    
    Checks:
    1. H(0) is Hermitian
    2. H(R) = H(-R)† for all R
    3. S(R) = S(-R)^T for all R
    4. Matrix dimensions are consistent
    
    Returns
    -------
    bool
        True if all checks pass
    """
    print("\n" + "=" * 70)
    print("VERIFICATION OF CONSTRUCTED MATRICES")
    print("=" * 70)
    
    all_passed = True
    
    # Check 1: Matrix consistency
    print("\n1. Checking matrix consistency...")
    if check_matrix_consistency(real_space_matrices):
        print("   ✓ All matrices have consistent dimensions")
    else:
        print("   ✗ Matrix dimension inconsistency detected!")
        all_passed = False
    
    # Check 2: Real-space symmetries using dedicated function
    print("\n2. Checking real-space symmetries...")
    symmetry_ok = verify_real_space_symmetry(
        real_space_matrices, 
        tolerance=tolerance,
        verbose=verbose
    )
    if symmetry_ok:
        print("   ✓ All real-space symmetries satisfied")
    else:
        print("   ✗ Real-space symmetry violations detected!")
        all_passed = False
    
    # Check 3: Detailed Hermiticity for H(0)
    origin = (0, 0, 0)
    if origin in real_space_matrices and 'H' in real_space_matrices[origin]:
        H_0 = real_space_matrices[origin]['H']
        diag_imag = np.max(np.abs(np.imag(np.diag(H_0))))
        print(f"\n3. H(0) diagonal imaginary part: {diag_imag:.2e}")
        if diag_imag > tolerance:
            print("   ✗ H(0) diagonal has non-zero imaginary parts!")
            all_passed = False
        else:
            print("   ✓ H(0) diagonal is real")
    
    # Summary
    print("\n" + "-" * 70)
    if all_passed:
        print("VERIFICATION PASSED: All physical requirements satisfied")
    else:
        print("VERIFICATION FAILED: Some requirements not satisfied")
    print("-" * 70)
    
    return all_passed


# ============================================================================
# Complete Test Workflow
# ============================================================================

def test_lcao_to_wannier90(lcao_file='--example', 
                           k_grid=(4, 4, 4),
                           num_wann=None,
                           seedname='test_material',
                           parallel=True,
                           verify=True):
    """
    Complete test of LCAO to Wannier90 workflow.
    
    Parameters
    ----------
    lcao_file : str
        Path to LCAO output file, or '--example' for synthetic data
    k_grid : tuple
        K-point grid dimensions
    num_wann : int, optional
        Number of Wannier functions
    seedname : str
        Output file prefix
    parallel : bool
        Use parallel processing
    verify : bool
        Run verification checks
    
    Returns
    -------
    engine : Wannier90Engine or None
        The engine instance if successful
    results : dict
        Results dictionary
    """
    print("\n" + "=" * 70)
    print("LCAO TO WANNIER90 - COMPLETE WORKFLOW TEST")
    print("=" * 70)
    
    try:
        # Step 1: Parse LCAO file
        matrices, lattice_vectors = parse_lcao_file(lcao_file)
        
        # Step 2: Prepare real-space matrices
        real_space_matrices, N_basis = prepare_real_space_data(
            matrices, lattice_vectors, verbose=True
        )
        
        # Step 3: Verify matrices
        if verify:
            verify_ok = verify_matrices(real_space_matrices, verbose=True)
            if not verify_ok:
                print("\nWARNING: Matrix verification failed. Results may be unreliable.")
        
        # Step 4: Print summary
        print_matrix_summary(real_space_matrices)
        
        # Step 5: Run Wannier90 engine (if available)
        if FULL_PACKAGE:
            print("\n" + "=" * 70)
            print("RUNNING WANNIER90 ENGINE")
            print("=" * 70)
            
            # Determine num_wann if not specified
            if num_wann is None:
                num_wann = 2 * N_basis  # Full basis
                print(f"  Using full basis: num_wann = {num_wann}")
            
            engine = Wannier90Engine(
                real_space_matrices=real_space_matrices,
                k_grid=k_grid,
                lattice_vectors=lattice_vectors,
                num_wann=num_wann,
                seedname=seedname
            )
            
            results = engine.run(parallel=parallel, verify=verify)
            
            print("\n" + "=" * 70)
            print("WORKFLOW COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Output files created:")
            print(f"  - {seedname}.eig")
            print(f"  - {seedname}.amn")
            print(f"  - {seedname}.mmn")
            
            return engine, results
        else:
            print("\n" + "=" * 70)
            print("PARSING AND MATRIX CONSTRUCTION COMPLETED")
            print("=" * 70)
            print("Note: Full Wannier90Engine not available.")
            print("      Matrices are ready for external processing.")
            
            return None, {'real_space_matrices': real_space_matrices, 
                         'N_basis': N_basis,
                         'lattice_vectors': lattice_vectors}
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test LCAO parser to Wannier90 workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_lcao_parser.py --example
  python test_lcao_parser.py my_crystal_output.out
  python test_lcao_parser.py my_crystal_output.out --k-grid 8 8 8 --num-wann 20
        """
    )
    parser.add_argument('lcao_file', nargs='?', default='--example',
                       help='LCAO output file (use --example for synthetic data)')
    parser.add_argument('--k-grid', nargs=3, type=int, default=[4, 4, 4],
                       help='K-point grid (default: 4 4 4)')
    parser.add_argument('--num-wann', type=int, default=None,
                       help='Number of Wannier functions (default: full basis)')
    parser.add_argument('--seedname', default='test_material',
                       help='Output file prefix (default: test_material)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification checks')
    
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
            parallel=not args.no_parallel,
            verify=not args.no_verify
        )
        
        if 'error' in results:
            return 1
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())