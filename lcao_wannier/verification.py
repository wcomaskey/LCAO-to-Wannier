"""
Verification Module

This module contains functions for verifying the numerical accuracy
and physical correctness of computed quantities.
"""

import numpy as np
import warnings
from typing import List, Tuple


def is_hermitian(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is Hermitian.
    
    Parameters
    ----------
    matrix : ndarray
        Matrix to check
    tol : float
        Tolerance for comparison
    
    Returns
    -------
    bool
        True if matrix is Hermitian within tolerance
    """
    return np.allclose(matrix, matrix.conj().T, atol=tol)


def verify_hermiticity(
    H_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    tol: float = 1e-10,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Verify that H(k) and S(k) are Hermitian for all k-points.
    
    Parameters
    ----------
    H_k_list : list of ndarrays
        Hamiltonian matrices for all k-points
    S_k_list : list of ndarrays
        Overlap matrices for all k-points
    tol : float
        Tolerance for Hermiticity check
    verbose : bool
        Whether to print results
    
    Returns
    -------
    max_H_deviation : float
        Maximum deviation from Hermiticity for H(k)
    max_S_deviation : float
        Maximum deviation from Hermiticity for S(k)
    """
    if verbose:
        print("\nVerifying Hermiticity of H(k) and S(k)...")
    
    max_H_deviation = 0.0
    max_S_deviation = 0.0
    
    for k_idx in range(len(H_k_list)):
        H_k = H_k_list[k_idx]
        S_k = S_k_list[k_idx]
        
        H_deviation = np.max(np.abs(H_k - H_k.conj().T))
        S_deviation = np.max(np.abs(S_k - S_k.conj().T))
        
        max_H_deviation = max(max_H_deviation, H_deviation)
        max_S_deviation = max(max_S_deviation, S_deviation)
    
    if verbose:
        print(f"  Max H(k) Hermiticity deviation: {max_H_deviation:.2e}")
        print(f"  Max S(k) Hermiticity deviation: {max_S_deviation:.2e}")
    
    if max_H_deviation > tol:
        warnings.warn(f"H(k) is not Hermitian within tolerance {tol}")
    if max_S_deviation > tol:
        warnings.warn(f"S(k) is not Hermitian within tolerance {tol}")
    
    return max_H_deviation, max_S_deviation


def verify_orthonormality(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_wann: int,
    num_check: int = 5,
    tol: float = 1e-8,
    verbose: bool = True
) -> float:
    """
    Verify that eigenvectors satisfy C†(k) S(k) C(k) ≈ I.
    
    Parameters
    ----------
    eigenvectors_list : list of ndarrays
        Eigenvectors for all k-points
    S_k_list : list of ndarrays
        Overlap matrices for all k-points
    num_wann : int
        Number of Wannier functions
    num_check : int
        Number of k-points to check
    tol : float
        Tolerance for orthonormality check
    verbose : bool
        Whether to print results
    
    Returns
    -------
    max_deviation : float
        Maximum deviation from identity matrix
    """
    if verbose:
        print(f"\nVerifying orthonormality at {num_check} k-points...")
    
    num_kpoints = len(eigenvectors_list)
    check_indices = np.linspace(0, num_kpoints - 1, num_check, dtype=int)
    
    max_deviation = 0.0
    
    for k_idx in check_indices:
        C_k = eigenvectors_list[k_idx]
        S_k = S_k_list[k_idx]
        
        # Compute C†(k) S(k) C(k)
        overlap = C_k.conj().T @ S_k @ C_k
        
        # Should be identity matrix
        identity = np.eye(num_wann)
        deviation = np.max(np.abs(overlap - identity))
        
        max_deviation = max(max_deviation, deviation)
        
        if verbose:
            print(f"  k-point {k_idx}: max deviation from identity = {deviation:.2e}")
        
        if deviation > tol:
            warnings.warn(f"Orthonormality check failed at k-point {k_idx}")
    
    return max_deviation


def verify_eigenvalue_sorting(
    eigenvalues_list: List[np.ndarray],
    verbose: bool = True
) -> bool:
    """
    Verify that eigenvalues are sorted in ascending order at each k-point.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for all k-points
    verbose : bool
        Whether to print results
    
    Returns
    -------
    bool
        True if all eigenvalues are properly sorted
    """
    if verbose:
        print("\nVerifying eigenvalue sorting...")
    
    all_sorted = True
    
    for k_idx, eigenvalues in enumerate(eigenvalues_list):
        if not np.all(eigenvalues[:-1] <= eigenvalues[1:]):
            all_sorted = False
            if verbose:
                print(f"  Warning: Eigenvalues at k-point {k_idx} are not sorted")
    
    if verbose and all_sorted:
        print("  ✓ All eigenvalues are properly sorted")
    
    return all_sorted


def compute_band_gaps(
    eigenvalues_list: List[np.ndarray],
    num_wann: int
) -> Tuple[float, float, int]:
    """
    Compute the minimum direct band gap across all k-points.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for all k-points
    num_wann : int
        Number of Wannier functions
    
    Returns
    -------
    min_gap : float
        Minimum direct band gap
    max_gap : float
        Maximum direct band gap
    k_idx_min : int
        k-point index where minimum gap occurs
    """
    gaps = []
    
    for k_idx, eigenvalues in enumerate(eigenvalues_list):
        if len(eigenvalues) >= 2:
            # Gap between highest and lowest band (simplified)
            gap = eigenvalues[-1] - eigenvalues[0]
            gaps.append((gap, k_idx))
    
    if gaps:
        min_gap, k_idx_min = min(gaps)
        max_gap, k_idx_max = max(gaps)
        return min_gap, max_gap, k_idx_min
    else:
        return 0.0, 0.0, 0


def verify_energy_range(
    eigenvalues_list: List[np.ndarray],
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Check the energy range of computed eigenvalues.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for all k-points
    verbose : bool
        Whether to print results
    
    Returns
    -------
    E_min : float
        Minimum eigenvalue across all k-points
    E_max : float
        Maximum eigenvalue across all k-points
    """
    all_eigenvalues = np.concatenate(eigenvalues_list)
    E_min = np.min(all_eigenvalues.real)
    E_max = np.max(all_eigenvalues.real)
    
    if verbose:
        print("\nEnergy range:")
        print(f"  Minimum eigenvalue: {E_min:.6f}")
        print(f"  Maximum eigenvalue: {E_max:.6f}")
        print(f"  Energy span: {E_max - E_min:.6f}")
    
    return E_min, E_max


def run_all_verifications(
    eigenvalues_list: List[np.ndarray],
    eigenvectors_list: List[np.ndarray],
    H_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_wann: int,
    verbose: bool = True
) -> dict:
    """
    Run all verification checks and return results.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for all k-points
    eigenvectors_list : list of ndarrays
        Eigenvectors for all k-points
    H_k_list : list of ndarrays
        Hamiltonian matrices for all k-points
    S_k_list : list of ndarrays
        Overlap matrices for all k-points
    num_wann : int
        Number of Wannier functions
    verbose : bool
        Whether to print results
    
    Returns
    -------
    dict
        Dictionary containing all verification results
    """
    results = {}
    
    # Hermiticity check
    max_H_dev, max_S_dev = verify_hermiticity(H_k_list, S_k_list, verbose=verbose)
    results['hermiticity'] = {'H_deviation': max_H_dev, 'S_deviation': max_S_dev}
    
    # Orthonormality check
    max_ortho_dev = verify_orthonormality(
        eigenvectors_list, S_k_list, num_wann, verbose=verbose
    )
    results['orthonormality'] = {'max_deviation': max_ortho_dev}
    
    # Eigenvalue sorting check
    sorting_ok = verify_eigenvalue_sorting(eigenvalues_list, verbose=verbose)
    results['eigenvalue_sorting'] = {'sorted': sorting_ok}
    
    # Energy range
    E_min, E_max = verify_energy_range(eigenvalues_list, verbose=verbose)
    results['energy_range'] = {'E_min': E_min, 'E_max': E_max}
    
    return results
