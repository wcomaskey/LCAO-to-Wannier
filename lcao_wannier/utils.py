"""
Utility Functions Module

This module contains helper functions for data format conversion,
matrix organization, and rigorous symmetry checking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ==============================
# Data Preparation & Conversion
# ==============================

def prepare_real_space_matrices(
    H_full_list: List[Tuple[np.ndarray, np.ndarray]],
    S_full_list: List[Tuple[np.ndarray, np.ndarray]],
    lattice_vectors: np.ndarray
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Convert from parsed (Cartesian) format to the (Integer) lattice format
    expected by the main engine.
    
    Parameters
    ----------
    H_full_list : list
        List of (R_cartesian, H_matrix) from parser
    S_full_list : list
        List of (R_cartesian, S_matrix) from parser
    lattice_vectors : ndarray
        3x3 matrix of real-space lattice vectors
    
    Returns
    -------
    real_space_matrices : dict
        Maps (n1, n2, n3) -> {'H': H_matrix, 'S': S_matrix}
    """
    real_space_matrices = {}
    
    # Process Hamiltonian matrices
    for R_cartesian, H_matrix in H_full_list:
        # Solve R_cart = n * A to find integers n
        # Rounding is necessary due to floating point noise
        R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
        R_tuple = tuple(R_integer)
        
        if R_tuple not in real_space_matrices:
            real_space_matrices[R_tuple] = {}
        real_space_matrices[R_tuple]['H'] = H_matrix
    
    # Process overlap matrices
    for R_cartesian, S_matrix in S_full_list:
        R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
        R_tuple = tuple(R_integer)
        
        if R_tuple not in real_space_matrices:
            real_space_matrices[R_tuple] = {}
        real_space_matrices[R_tuple]['S'] = S_matrix
    
    return real_space_matrices


def organize_matrices_by_lattice_vector(
    matrices: List[dict]
) -> Tuple[Dict[Tuple[int, int, int], Dict[str, np.ndarray]], 
           Dict[Tuple[int, int, int], np.ndarray]]:
    """
    Organize raw parsed matrix blocks by lattice vector and spin channel.
    
    This bridges the gap between 'parse_overlap_and_fock_matrices' and
    'create_spin_block_matrices'.
    """
    H_R_dict = {}
    S_R_dict = {}
    
    for matrix_info in matrices:
        matrix_type = matrix_info['type']
        lattice_vector = tuple(matrix_info['lattice_vector'])
        data = matrix_info['data']
        
        if data is None:
            continue
            
        if matrix_type == 'overlap':
            S_R_dict[lattice_vector] = data
        elif matrix_type == 'fock':
            spin_channel = matrix_info['spin_channel']
            if lattice_vector not in H_R_dict:
                H_R_dict[lattice_vector] = {}
            H_R_dict[lattice_vector][spin_channel] = data
    
    return H_R_dict, S_R_dict


def get_basis_size(matrices_dict: dict) -> int:
    """Get the basis size (N) from the matrices dictionary."""
    if not matrices_dict:
        return 0
    first_key = next(iter(matrices_dict))
    
    if 'H' in matrices_dict[first_key]:
        # H is 2N x 2N, we want N (spatial orbitals)
        return matrices_dict[first_key]['H'].shape[0] // 2
    elif 'ALPHA_ALPHA' in matrices_dict[first_key]:
        # Raw dict is N x N
        return matrices_dict[first_key]['ALPHA_ALPHA'].shape[0]
    return 0


# ==============================
# Verification & Diagnostics
# ==============================

def verify_matrix_symmetry(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    tolerance: float = 1e-10
) -> bool:
    """
    Verify that the constructed matrices satisfy the fundamental physical symmetries:
    1. H(0) is Hermitian.
    2. H(R) = H(-R)†
    3. S(R) = S(-R)^T
    
    Returns True if all checks pass.
    """
    print("\n" + "-" * 60)
    print("VERIFYING MATRIX SYMMETRY AND HERMITICITY")
    print("-" * 60)
    
    all_passed = True
    max_error_H = 0.0
    max_error_S = 0.0
    
    # Check Origin
    origin = (0, 0, 0)
    if origin in real_space_matrices:
        mats = real_space_matrices[origin]
        if 'H' in mats:
            # Check H(0) == H(0)†
            diff = np.max(np.abs(mats['H'] - mats['H'].conj().T))
            if diff > tolerance:
                print(f"FAIL: Origin H(0) is not Hermitian. Max Diff: {diff:.2e}")
                all_passed = False
            max_error_H = max(max_error_H, diff)
            
    # Check Pairs
    checked_R = set()
    for R in real_space_matrices:
        if R == (0, 0, 0) or R in checked_R:
            continue
            
        minus_R = tuple(-x for x in R)
        if minus_R not in real_space_matrices:
            print(f"WARNING: pair {minus_R} missing for {R}")
            continue
            
        # Check Hamiltonian: H(R) - H(-R)† == 0
        if 'H' in real_space_matrices[R] and 'H' in real_space_matrices[minus_R]:
            H_R = real_space_matrices[R]['H']
            H_mR = real_space_matrices[minus_R]['H']
            
            diff = np.max(np.abs(H_R - H_mR.conj().T))
            max_error_H = max(max_error_H, diff)
            
            if diff > tolerance:
                print(f"FAIL H: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        # Check Overlap: S(R) - S(-R)^T == 0 (Transpose only, S is real)
        if 'S' in real_space_matrices[R] and 'S' in real_space_matrices[minus_R]:
            S_R = real_space_matrices[R]['S']
            S_mR = real_space_matrices[minus_R]['S']
            
            diff = np.max(np.abs(S_R - S_mR.T))
            max_error_S = max(max_error_S, diff)
            
            if diff > tolerance:
                print(f"FAIL S: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        checked_R.add(R)
        checked_R.add(minus_R)

    print(f"Max Symmetry Error H: {max_error_H:.2e}")
    print(f"Max Symmetry Error S: {max_error_S:.2e}")
    
    if all_passed:
        print("SUCCESS: All matrix symmetries are satisfied.")
    else:
        print("FAILURE: Matrix symmetries violated.")
        
    return all_passed


def check_matrix_consistency(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> bool:
    """Check that all matrices have consistent dimensions (square and equal size)."""
    sizes = []
    
    for R_tuple, matrices in real_space_matrices.items():
        if 'H' in matrices:
            H_shape = matrices['H'].shape
            if H_shape[0] != H_shape[1]:
                print(f"Warning: H matrix at R={R_tuple} is not square")
                return False
            sizes.append(H_shape[0])
        
        if 'S' in matrices:
            S_shape = matrices['S'].shape
            if S_shape[0] != S_shape[1]:
                print(f"Warning: S matrix at R={R_tuple} is not square")
                return False
            sizes.append(S_shape[0])
    
    if len(set(sizes)) > 1:
        print(f"Warning: Inconsistent matrix sizes: {set(sizes)}")
        return False
    
    return True


# ==============================
# Reporting
# ==============================

def print_matrix_summary(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> None:
    """Print a summary of the real-space matrices."""
    print("\n" + "=" * 70)
    print("Real-Space Matrices Summary")
    print("=" * 70)
    
    num_R_vectors = len(real_space_matrices)
    print(f"Number of R-vectors: {num_R_vectors}")
    
    if num_R_vectors > 0:
        # Get matrix size
        first_key = next(iter(real_space_matrices))
        if 'H' in real_space_matrices[first_key]:
            dim = real_space_matrices[first_key]['H'].shape[0]
            print(f"Hamiltonian Dimension: {dim} × {dim}")
    
    # List R-vectors
    print("\nR-vectors:")
    for R_tuple in sorted(real_space_matrices.keys()):
        has_H = 'H' in real_space_matrices[R_tuple]
        has_S = 'S' in real_space_matrices[R_tuple]
        print(f"  {R_tuple}: H={'✓' if has_H else '✗'}, S={'✓' if has_S else '✗'}")
    
    print("=" * 70)


def print_calculation_info(
    num_kpoints: int,
    k_grid: Tuple[int, int, int],
    num_orbitals: int,
    num_wann: int
) -> None:
    """Print information about the calculation parameters and memory."""
    print("\n" + "=" * 70)
    print("Calculation Information")
    print("=" * 70)
    print(f"K-point grid: {k_grid[0]} × {k_grid[1]} × {k_grid[2]}")
    print(f"Number of k-points: {num_kpoints}")
    print(f"Number of orbitals: {num_orbitals}")
    print(f"Number of Wannier functions: {num_wann}")
    
    # Memory estimate
    bytes_per_complex = 16
    matrices_mb = 2 * num_kpoints * num_orbitals**2 * bytes_per_complex / 1e6
    eigenvalues_mb = num_kpoints * num_wann * 8 / 1e6
    eigenvectors_mb = num_kpoints * num_orbitals * num_wann * bytes_per_complex / 1e6
    total_mb = matrices_mb + eigenvalues_mb + eigenvectors_mb
    
    print(f"\nEstimated memory usage:")
    print(f"  Matrices (H, S): {matrices_mb:.1f} MB")
    print(f"  Eigenvalues:     {eigenvalues_mb:.1f} MB")
    print(f"  Eigenvectors:    {eigenvectors_mb:.1f} MB")
    print(f"  Total:           {total_mb:.1f} MB")
    print("=" * 70)