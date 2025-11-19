"""
Utility Functions Module

This module contains helper functions for data format conversion
and other utility operations.
"""

import numpy as np
from typing import Dict, List, Tuple


def prepare_real_space_matrices(
    H_full_list: List[Tuple[np.ndarray, np.ndarray]],
    S_full_list: List[Tuple[np.ndarray, np.ndarray]],
    lattice_vectors: np.ndarray
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Convert from parsed format to the format expected by Wannier90Engine.
    
    Converts from (R_cartesian, matrix) format to (R_integer, matrix) format.
    
    Parameters
    ----------
    H_full_list : list of tuples
        List of (R_cartesian, H_matrix) tuples from parsing
    S_full_list : list of tuples
        List of (R_cartesian, S_matrix) tuples from parsing
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are vectors)
    
    Returns
    -------
    real_space_matrices : dict
        Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
        where (R1, R2, R3) are integer lattice coordinates
    
    Examples
    --------
    >>> H_list = [(R_cart1, H1), (R_cart2, H2)]
    >>> S_list = [(R_cart1, S1), (R_cart2, S2)]
    >>> matrices = prepare_real_space_matrices(H_list, S_list, lattice_vectors)
    """
    real_space_matrices = {}
    
    # Process Hamiltonian matrices
    for R_cartesian, H_matrix in H_full_list:
        # Convert Cartesian coordinates to lattice coordinates
        # Solve: R_cartesian = R_integer @ lattice_vectors
        R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
        R_tuple = tuple(R_integer)
        
        if R_tuple not in real_space_matrices:
            real_space_matrices[R_tuple] = {}
        real_space_matrices[R_tuple]['H'] = H_matrix
    
    # Process overlap matrices
    for R_cartesian, S_matrix in S_full_list:
        # Convert Cartesian coordinates to lattice coordinates
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
    Organize parsed matrices by lattice vector.
    
    Separates overlap and Fock matrices and organizes them by
    their lattice vectors and spin channels.
    
    Parameters
    ----------
    matrices : list of dict
        Parsed matrix information from parser
    
    Returns
    -------
    H_R_dict : dict
        Maps (R1, R2, R3) -> {spin_channel: H_matrix}
    S_R_dict : dict
        Maps (R1, R2, R3) -> S_matrix
    """
    H_R_dict = {}
    S_R_dict = {}
    
    for matrix_info in matrices:
        matrix_type = matrix_info['type']
        lattice_vector = tuple(matrix_info['lattice_vector'])
        data = matrix_info['data']
        
        if matrix_type == 'overlap':
            S_R_dict[lattice_vector] = data
        elif matrix_type == 'fock':
            spin_channel = matrix_info['spin_channel']
            if lattice_vector not in H_R_dict:
                H_R_dict[lattice_vector] = {}
            H_R_dict[lattice_vector][spin_channel] = data
    
    return H_R_dict, S_R_dict


def get_basis_size(matrices_dict: dict) -> int:
    """
    Get the basis size from the matrices dictionary.
    
    Parameters
    ----------
    matrices_dict : dict
        Dictionary containing matrices
    
    Returns
    -------
    int
        Number of basis functions
    """
    first_key = next(iter(matrices_dict))
    return matrices_dict[first_key].shape[0]


def check_matrix_consistency(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> bool:
    """
    Check that all matrices have consistent dimensions.
    
    Parameters
    ----------
    real_space_matrices : dict
        Dictionary of real-space matrices
    
    Returns
    -------
    bool
        True if all matrices are consistent
    """
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


def print_matrix_summary(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> None:
    """
    Print a summary of the real-space matrices.
    
    Parameters
    ----------
    real_space_matrices : dict
        Dictionary of real-space matrices
    """
    print("\n" + "=" * 70)
    print("Real-Space Matrices Summary")
    print("=" * 70)
    
    num_R_vectors = len(real_space_matrices)
    print(f"Number of R-vectors: {num_R_vectors}")
    
    # Get matrix size
    first_key = next(iter(real_space_matrices))
    num_orbitals = real_space_matrices[first_key]['H'].shape[0]
    print(f"Matrix size: {num_orbitals} × {num_orbitals}")
    
    # List R-vectors
    print("\nR-vectors:")
    for R_tuple in sorted(real_space_matrices.keys()):
        has_H = 'H' in real_space_matrices[R_tuple]
        has_S = 'S' in real_space_matrices[R_tuple]
        print(f"  {R_tuple}: H={'✓' if has_H else '✗'}, S={'✓' if has_S else '✗'}")
    
    print("=" * 70)


def estimate_memory_usage(
    num_kpoints: int,
    num_orbitals: int,
    num_wann: int
) -> dict:
    """
    Estimate memory usage for the calculation.
    
    Parameters
    ----------
    num_kpoints : int
        Number of k-points
    num_orbitals : int
        Number of orbitals
    num_wann : int
        Number of Wannier functions
    
    Returns
    -------
    dict
        Dictionary with memory estimates in MB
    """
    bytes_per_complex = 16  # complex128
    
    # H(k) and S(k) for all k-points
    matrices_mb = 2 * num_kpoints * num_orbitals**2 * bytes_per_complex / 1e6
    
    # Eigenvalues for all k-points
    eigenvalues_mb = num_kpoints * num_wann * 8 / 1e6
    
    # Eigenvectors for all k-points
    eigenvectors_mb = num_kpoints * num_orbitals * num_wann * bytes_per_complex / 1e6
    
    total_mb = matrices_mb + eigenvalues_mb + eigenvectors_mb
    
    return {
        'matrices': matrices_mb,
        'eigenvalues': eigenvalues_mb,
        'eigenvectors': eigenvectors_mb,
        'total': total_mb
    }


def print_calculation_info(
    num_kpoints: int,
    k_grid: Tuple[int, int, int],
    num_orbitals: int,
    num_wann: int
) -> None:
    """
    Print information about the calculation.
    
    Parameters
    ----------
    num_kpoints : int
        Number of k-points
    k_grid : tuple of 3 ints
        K-point grid dimensions
    num_orbitals : int
        Number of orbitals
    num_wann : int
        Number of Wannier functions
    """
    print("\n" + "=" * 70)
    print("Calculation Information")
    print("=" * 70)
    print(f"K-point grid: {k_grid[0]} × {k_grid[1]} × {k_grid[2]}")
    print(f"Number of k-points: {num_kpoints}")
    print(f"Number of orbitals: {num_orbitals}")
    print(f"Number of Wannier functions: {num_wann}")
    
    # Memory estimate
    mem = estimate_memory_usage(num_kpoints, num_orbitals, num_wann)
    print(f"\nEstimated memory usage:")
    print(f"  Matrices (H, S): {mem['matrices']:.1f} MB")
    print(f"  Eigenvalues: {mem['eigenvalues']:.1f} MB")
    print(f"  Eigenvectors: {mem['eigenvectors']:.1f} MB")
    print(f"  Total: {mem['total']:.1f} MB")
    print("=" * 70)
