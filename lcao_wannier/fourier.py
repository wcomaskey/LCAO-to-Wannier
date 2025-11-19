"""
Fourier Transform Module

This module contains functions for Fourier transforming real-space matrices
to k-space using the phase factor e^(i 2π k·R).
"""

import numpy as np
from typing import Dict, Tuple


def fourier_transform_to_kspace(
    k_point: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier transform real-space H(R) and S(R) matrices to k-space.
    
    Computes:
        H(k) = Σ_R e^(i 2π k·R) H(R)
        S(k) = Σ_R e^(i 2π k·R) S(R)
    
    Parameters
    ----------
    k_point : ndarray of shape (3,)
        k-point in fractional coordinates
    real_space_matrices : dict
        Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
        where H_matrix and S_matrix are complex ndarrays
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are vectors)
    
    Returns
    -------
    H_k : ndarray
        Hamiltonian in k-space, shape (num_orbitals, num_orbitals)
    S_k : ndarray
        Overlap matrix in k-space, shape (num_orbitals, num_orbitals)
    
    Notes
    -----
    The phase factor is computed as exp(i 2π k·R) where k is in fractional
    coordinates and R is in integer lattice coordinates.
    
    Examples
    --------
    >>> k = np.array([0.25, 0.25, 0.25])
    >>> H_k, S_k = fourier_transform_to_kspace(k, real_space_matrices, lattice_vectors)
    """
    # Get matrix dimensions from first entry
    first_key = next(iter(real_space_matrices))
    num_orbitals = real_space_matrices[first_key]['H'].shape[0]
    
    # Initialize k-space matrices
    H_k = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
    S_k = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
    
    # Loop over all R vectors and accumulate Fourier sum
    for R_tuple, matrices in real_space_matrices.items():
        R = np.array(R_tuple)
        
        # Calculate phase factor: e^(i 2π k·R)
        # k is in fractional coordinates, R is in integer lattice coordinates
        phase = np.exp(2j * np.pi * np.dot(k_point, R))
        
        # Accumulate the Fourier sum
        H_k += phase * matrices['H']
        S_k += phase * matrices['S']
    
    return H_k, S_k


def inverse_fourier_transform(
    k_points: np.ndarray,
    H_k_list: list,
    S_k_list: list,
    R_vectors: list,
    k_grid: Tuple[int, int, int]
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Inverse Fourier transform from k-space to real-space (optional utility).
    
    Computes:
        H(R) = (1/N_k) Σ_k e^(-i 2π k·R) H(k)
        S(R) = (1/N_k) Σ_k e^(-i 2π k·R) S(k)
    
    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        k-points in fractional coordinates
    H_k_list : list of ndarrays
        List of Hamiltonian matrices at each k-point
    S_k_list : list of ndarrays
        List of overlap matrices at each k-point
    R_vectors : list of tuples
        List of R vectors to compute
    k_grid : tuple of 3 ints
        Dimensions of the k-point grid
    
    Returns
    -------
    real_space_matrices : dict
        Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
    """
    num_kpoints = len(k_points)
    num_orbitals = H_k_list[0].shape[0]
    real_space_matrices = {}
    
    for R_tuple in R_vectors:
        R = np.array(R_tuple)
        H_R = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        S_R = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        
        for k_idx, k_point in enumerate(k_points):
            # Phase factor: e^(-i 2π k·R)
            phase = np.exp(-2j * np.pi * np.dot(k_point, R))
            
            H_R += phase * H_k_list[k_idx]
            S_R += phase * S_k_list[k_idx]
        
        # Normalize by number of k-points
        H_R /= num_kpoints
        S_R /= num_kpoints
        
        real_space_matrices[R_tuple] = {'H': H_R, 'S': S_R}
    
    return real_space_matrices


def compute_phase_factors(
    k_points: np.ndarray,
    R_vectors: list
) -> np.ndarray:
    """
    Pre-compute phase factors for all k-points and R vectors.
    
    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        k-points in fractional coordinates
    R_vectors : list of tuples
        List of (R1, R2, R3) tuples
    
    Returns
    -------
    phase_factors : ndarray of shape (num_kpoints, num_R_vectors)
        Pre-computed phase factors
    """
    num_kpoints = len(k_points)
    num_R = len(R_vectors)
    phase_factors = np.zeros((num_kpoints, num_R), dtype=np.complex128)
    
    for k_idx, k_point in enumerate(k_points):
        for R_idx, R_tuple in enumerate(R_vectors):
            R = np.array(R_tuple)
            phase_factors[k_idx, R_idx] = np.exp(2j * np.pi * np.dot(k_point, R))
    
    return phase_factors
