"""
Verification Module

This module contains functions for verifying the numerical accuracy
and physical correctness of computed quantities, including 
real-space matrix symmetries and k-space properties.
"""

import numpy as np
import warnings
from typing import List, Tuple, Dict

# ==============================
# Basic Utility
# ==============================

def is_hermitian(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is Hermitian."""
    return np.allclose(matrix, matrix.conj().T, atol=tol)


# ==============================
# Real-Space Verification
# ==============================

def verify_real_space_symmetry(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    tolerance: float = 1e-10,
    verbose: bool = True
) -> bool:
    """
    Verify that the constructed real-space matrices satisfy fundamental physical symmetries:
    1. H(0) is Hermitian.
    2. H(R) = H(-R)†
    3. S(R) = S(-R)^T
    
    Parameters
    ----------
    real_space_matrices : dict
        Dictionary mapping (R_int) -> {'H': matrix, 'S': matrix}
    tolerance : float
        Numerical tolerance for checks
    verbose : bool
        Whether to print detailed output
    
    Returns
    -------
    bool
        True if all checks pass.
    """
    if verbose:
        print("\n" + "-" * 60)
        print("VERIFYING REAL-SPACE MATRIX SYMMETRIES")
        print("-" * 60)
    
    all_passed = True
    max_error_H = 0.0
    max_error_S = 0.0
    
    # 1. Check Origin
    origin = (0, 0, 0)
    if origin in real_space_matrices:
        mats = real_space_matrices[origin]
        if 'H' in mats:
            # Check H(0) == H(0)†
            diff = np.max(np.abs(mats['H'] - mats['H'].conj().T))
            if diff > tolerance:
                if verbose:
                    print(f"FAIL: Origin H(0) is not Hermitian. Max Diff: {diff:.2e}")
                all_passed = False
            max_error_H = max(max_error_H, diff)
            
    # 2. Check Pairs
    checked_R = set()
    for R in real_space_matrices:
        if R == (0, 0, 0) or R in checked_R:
            continue
            
        minus_R = tuple(-x for x in R)
        
        # Check if pair exists
        if minus_R not in real_space_matrices:
            if verbose:
                print(f"WARNING: pair {minus_R} missing for {R}")
            continue
            
        # Check Hamiltonian: H(R) - H(-R)† == 0
        if 'H' in real_space_matrices[R] and 'H' in real_space_matrices[minus_R]:
            H_R = real_space_matrices[R]['H']
            H_mR = real_space_matrices[minus_R]['H']
            
            diff = np.max(np.abs(H_R - H_mR.conj().T))
            max_error_H = max(max_error_H, diff)
            
            if diff > tolerance:
                if verbose:
                    print(f"FAIL H: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        # Check Overlap: S(R) - S(-R)^T == 0 (Transpose only, S is real)
        if 'S' in real_space_matrices[R] and 'S' in real_space_matrices[minus_R]:
            S_R = real_space_matrices[R]['S']
            S_mR = real_space_matrices[minus_R]['S']
            
            diff = np.max(np.abs(S_R - S_mR.T))
            max_error_S = max(max_error_S, diff)
            
            if diff > tolerance:
                if verbose:
                    print(f"FAIL S: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        checked_R.add(R)
        checked_R.add(minus_R)

    if verbose:
        print(f"  Max Real-Space H(R) Symmetry Error: {max_error_H:.2e}")
        print(f"  Max Real-Space S(R) Symmetry Error: {max_error_S:.2e}")
    
    if max_error_H > tolerance:
        warnings.warn(f"Real-space H(R) symmetry violated (Max Err: {max_error_H:.2e})")
    if max_error_S > tolerance:
        warnings.warn(f"Real-space S(R) symmetry violated (Max Err: {max_error_S:.2e})")
        
    return all_passed


# ==============================
# K-Space Verification
# ==============================

def verify_hermiticity(
    H_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    tol: float = 1e-10,
    verbose: bool = True
) -> Tuple[float, float]:
    """Verify that H(k) and S(k) are Hermitian for all k-points."""
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
    """Verify that eigenvectors satisfy C†(k) S(k) C(k) ≈ I."""
    if verbose:
        print(f"\nVerifying orthonormality at {num_check} k-points...")
    
    num_kpoints = len(eigenvectors_list)
    # Ensure we don't try to check more points than exist
    actual_checks = min(num_check, num_kpoints)
    check_indices = np.linspace(0, num_kpoints - 1, actual_checks, dtype=int)
    
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
    """Verify that eigenvalues are sorted in ascending order at each k-point."""
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
    """Compute the minimum direct band gap across all k-points."""
    gaps = []
    
    for k_idx, eigenvalues in enumerate(eigenvalues_list):
        if len(eigenvalues) >= 2:
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
    """Check the energy range of computed eigenvalues."""
    if not eigenvalues_list:
        return 0.0, 0.0
        
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
    """Run all k-space verification checks and return results."""
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