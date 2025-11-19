"""
Eigenvalue Solver Module

This module contains functions for solving the generalized eigenvalue problem
H(k) C(k) = S(k) C(k) E(k) at each k-point.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Dict
from .fourier import fourier_transform_to_kspace


def solve_generalized_eigenvalue_problem(
    H_k: np.ndarray,
    S_k: np.ndarray,
    num_wann: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem H(k) C = S(k) C E.
    
    Uses scipy.linalg.eigh for Hermitian matrices, which is more stable
    and efficient than the general eigenvalue solver.
    
    Parameters
    ----------
    H_k : ndarray of shape (num_orbitals, num_orbitals)
        Hamiltonian matrix at k-point
    S_k : ndarray of shape (num_orbitals, num_orbitals)
        Overlap matrix at k-point
    num_wann : int, optional
        Number of lowest eigenvalues/eigenvectors to keep
        If None, keeps all
    
    Returns
    -------
    eigenvalues : ndarray of shape (num_wann,)
        Eigenvalues sorted in ascending order
    eigenvectors : ndarray of shape (num_orbitals, num_wann)
        Eigenvectors (columns), normalized such that C† S C = I
    
    Notes
    -----
    The generalized eigenvalue problem is:
        H(k) C(k) = S(k) C(k) E(k)
    
    scipy.linalg.eigh automatically sorts eigenvalues in ascending order
    and normalizes eigenvectors according to C† S C = I.
    """
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(H_k, S_k)
    
    # Keep only the lowest num_wann eigenvalues and eigenvectors
    if num_wann is not None:
        eigenvalues = eigenvalues[:num_wann]
        eigenvectors = eigenvectors[:, :num_wann]
    
    return eigenvalues, eigenvectors


def solve_kpoint(
    k_idx: int,
    k_point: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    num_wann: int
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem for a single k-point.
    
    This function is designed to be easily parallelizable.
    
    Parameters
    ----------
    k_idx : int
        Index of the k-point
    k_point : ndarray of shape (3,)
        k-point in fractional coordinates
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions (bands to keep)
    
    Returns
    -------
    k_idx : int
        k-point index (for sorting in parallel execution)
    eigenvalues : ndarray
        Eigenvalues for this k-point
    eigenvectors : ndarray
        Eigenvectors for this k-point
    H_k : ndarray
        Hamiltonian at this k-point (for verification)
    S_k : ndarray
        Overlap at this k-point (for verification)
    """
    # Fourier transform to k-space
    H_k, S_k = fourier_transform_to_kspace(k_point, real_space_matrices, lattice_vectors)
    
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = solve_generalized_eigenvalue_problem(H_k, S_k, num_wann)
    
    return k_idx, eigenvalues, eigenvectors, H_k, S_k


def solve_all_kpoints_sequential(
    k_points: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    num_wann: int
) -> Tuple[list, list, list, list]:
    """
    Solve eigenvalue problems at all k-points sequentially.
    
    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        All k-points in fractional coordinates
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions
    
    Returns
    -------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    H_k_list : list of ndarrays
        Hamiltonian matrices for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    """
    eigenvalues_list = []
    eigenvectors_list = []
    H_k_list = []
    S_k_list = []
    
    for k_idx in range(len(k_points)):
        k_point = k_points[k_idx]
        _, eigenvalues, eigenvectors, H_k, S_k = solve_kpoint(
            k_idx, k_point, real_space_matrices, lattice_vectors, num_wann
        )
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)
        H_k_list.append(H_k)
        S_k_list.append(S_k)
    
    return eigenvalues_list, eigenvectors_list, H_k_list, S_k_list


def solve_all_kpoints_parallel(
    k_points: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    num_wann: int,
    num_processes: int = None
) -> Tuple[list, list, list, list]:
    """
    Solve eigenvalue problems at all k-points in parallel.
    
    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        All k-points in fractional coordinates
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions
    num_processes : int, optional
        Number of parallel processes (default: use all CPUs)
    
    Returns
    -------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    H_k_list : list of ndarrays
        Hamiltonian matrices for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    """
    import multiprocessing as mp
    from functools import partial
    
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(k_points))
    
    # Create partial function with fixed parameters
    solve_func = partial(
        solve_kpoint,
        real_space_matrices=real_space_matrices,
        lattice_vectors=lattice_vectors,
        num_wann=num_wann
    )
    
    # Prepare arguments
    args = [(k_idx, k_points[k_idx]) for k_idx in range(len(k_points))]
    
    # Run parallel computation
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(solve_func, args)
    
    # Sort results by k_idx
    results.sort(key=lambda x: x[0])
    
    # Extract results
    eigenvalues_list = [r[1] for r in results]
    eigenvectors_list = [r[2] for r in results]
    H_k_list = [r[3] for r in results]
    S_k_list = [r[4] for r in results]
    
    return eigenvalues_list, eigenvectors_list, H_k_list, S_k_list
