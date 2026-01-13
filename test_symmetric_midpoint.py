#!/usr/bin/env python3
"""
Validation Test Script for Symmetric Midpoint MMN Implementation

This script runs 3 critical checks to validate the LCAO-Wannier MMN matrices:
1. Self-Consistency (Identity Check) - M(k,k) with b=0 should be Identity
2. Hermiticity (Symmetry Check) - M(k1,k2) should equal M(k2,k1)†
3. Boundedness (SVD Check) - Singular values should be reasonable

Usage:
    python test_symmetric_midpoint.py
"""

import numpy as np
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lcao_wannier.wannier90 import compute_mmn_matrix
from lcao_wannier.fourier import fourier_transform_to_kspace


def validate_mmn_implementation(kpoints, eigenvectors, real_space_matrices,
                                lattice_vectors, atom_pos, basis_map,
                                compute_mmn_func):
    """
    Runs the 3 Critical Checks for LCAO-Wannier MMN Matrices.

    Parameters
    ----------
    kpoints : ndarray of shape (num_kpoints, 3)
        K-points in fractional coordinates
    eigenvectors : list of ndarray
        Eigenvector matrices C(k) for each k-point
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    atom_pos : ndarray of shape (num_atoms, 3)
        Atomic positions in Cartesian coordinates
    basis_map : ndarray of shape (num_orbitals,)
        Maps each orbital to its atom index
    compute_mmn_func : callable
        The compute_mmn_matrix function to test

    Returns
    -------
    passed : bool
        True if all tests passed
    """
    print("=" * 60)
    print("=== STARTING SYMMETRIC MIDPOINT VALIDATION ===")
    print("=" * 60)

    all_passed = True

    # --- TEST 1: SELF-CONSISTENCY (Identity Check) ---
    print("\n[Test 1] Self-Consistency (b=0)")
    print("-" * 40)
    b_zero = np.zeros(3)

    # Compute M(k, k) with b=0
    M_self = compute_mmn_func(
        0, 0, kpoints, b_zero,
        real_space_matrices, lattice_vectors, atom_pos, basis_map,
        eigenvectors
    )

    # Should be Identity (approx)
    identity = np.eye(M_self.shape[0])
    dev_eye = np.max(np.abs(M_self - identity))
    diag_vals = np.diag(M_self)
    off_diag_max = np.max(np.abs(M_self - np.diag(diag_vals)))

    print(f"  Matrix size: {M_self.shape[0]} x {M_self.shape[1]}")
    print(f"  Deviation from Identity: {dev_eye:.4e}")
    print(f"  Diagonal values range: [{np.min(np.abs(diag_vals)):.6f}, {np.max(np.abs(diag_vals)):.6f}]")
    print(f"  Max off-diagonal element: {off_diag_max:.4e}")

    if dev_eye < 1e-10:
        print("  -> PASS")
    else:
        print("  -> FAIL (Check Normalization or b=0 logic)")
        all_passed = False

    # --- TEST 2: HERMITICITY (Symmetry Check) ---
    print("\n[Test 2] Hermiticity (Forward vs Backward)")
    print("-" * 40)

    # Pick k1 and neighbor k2 (ensure we have at least 2 k-points)
    if len(kpoints) < 2:
        print("  SKIP: Need at least 2 k-points for this test")
    else:
        idx1, idx2 = 0, 1
        k1, k2 = kpoints[idx1], kpoints[idx2]

        # Calculate physical b vector (Cartesian)
        # Assuming no boundary crossing for this simple test
        b_frac = k2 - k1
        b_cart = np.dot(b_frac, lattice_vectors)

        print(f"  k1 = {k1}")
        print(f"  k2 = {k2}")
        print(f"  b_frac = {b_frac}")
        print(f"  b_cart = {b_cart}")

        # Forward: <u_1 | u_2>
        M_12 = compute_mmn_func(
            idx1, idx2, kpoints, b_cart,
            real_space_matrices, lattice_vectors, atom_pos, basis_map,
            eigenvectors
        )

        # Backward: <u_2 | u_1> with -b
        M_21 = compute_mmn_func(
            idx2, idx1, kpoints, -b_cart,
            real_space_matrices, lattice_vectors, atom_pos, basis_map,
            eigenvectors
        )

        # M_12 should equal M_21^dagger
        herm_err = np.max(np.abs(M_12 - M_21.conj().T))
        print(f"  Hermiticity Error: {herm_err:.4e}")

        if herm_err < 1e-12:
            print("  -> PASS (Symmetric Midpoint Logic Works)")
        else:
            print("  -> FAIL (Midpoint or Phase Logic is Asymmetric)")
            all_passed = False

    # --- TEST 3: BOUNDEDNESS (SVD Check) ---
    print("\n[Test 3] Boundedness (Singular Values)")
    print("-" * 40)

    if len(kpoints) >= 2:
        s_vals = np.linalg.svd(M_12, compute_uv=False)
        print(f"  Singular Values: range [{np.min(s_vals):.4f}, {np.max(s_vals):.4f}]")
        print(f"  Condition number: {np.max(s_vals) / np.min(s_vals):.4f}")

        if np.min(s_vals) > 0.1 and np.max(s_vals) < 2.0:
            print("  -> PASS (Matrix is well-conditioned)")
        else:
            print("  -> WARNING (Matrix may be poorly conditioned, but not necessarily wrong)")
    else:
        print("  SKIP: Need M_12 from Test 2")

    print("\n" + "=" * 60)
    if all_passed:
        print("=== ALL VALIDATION TESTS PASSED ===")
    else:
        print("=== SOME TESTS FAILED ===")
    print("=" * 60)

    return all_passed


def create_test_system():
    """
    Create a simple test system for validation.

    Returns a minimal 2-atom system with s-orbitals for testing.
    """
    # Simple cubic lattice (4 Angstrom lattice constant)
    a = 4.0
    lattice_vectors = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ])

    # 2 atoms at origin and center
    atom_positions = np.array([
        [0.0, 0.0, 0.0],
        [a/2, a/2, a/2]
    ])

    # 2 s-orbitals, one per atom
    num_orbitals = 2
    basis_atom_map = np.array([0, 1])

    # Create simple real-space matrices (minimal set)
    # Include R = (0,0,0), (1,0,0), (-1,0,0), etc.
    real_space_matrices = {}

    # On-site terms
    H_onsite = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    S_onsite = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.complex128)
    real_space_matrices[(0, 0, 0)] = {'H': H_onsite, 'S': S_onsite}

    # Hopping terms (simplified)
    t = -0.5  # hopping parameter
    s_hop = 0.1  # overlap hopping

    for R in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        H_hop = np.array([[0.0, t], [t, 0.0]], dtype=np.complex128)
        S_hop = np.array([[0.0, s_hop], [s_hop, 0.0]], dtype=np.complex128)
        real_space_matrices[R] = {'H': H_hop, 'S': S_hop}

    # K-points (simple 2x2x2 grid)
    kpoints = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])

    # Compute eigenvectors at each k-point
    eigenvectors = []
    for k in kpoints:
        H_k, S_k = fourier_transform_to_kspace(k, real_space_matrices, lattice_vectors)

        # Solve generalized eigenvalue problem: H @ C = E @ S @ C
        eigenvalues, C = np.linalg.eigh(np.linalg.solve(S_k, H_k))

        # Normalize eigenvectors to satisfy C† @ S @ C = I
        for i in range(C.shape[1]):
            norm = np.sqrt(C[:, i].conj() @ S_k @ C[:, i])
            C[:, i] /= norm

        eigenvectors.append(C)

    return {
        'kpoints': kpoints,
        'eigenvectors': eigenvectors,
        'real_space_matrices': real_space_matrices,
        'lattice_vectors': lattice_vectors,
        'atom_positions': atom_positions,
        'basis_atom_map': basis_atom_map
    }


def main():
    """Run validation tests."""
    print("\n" + "=" * 60)
    print("  Symmetric Midpoint Method - Validation Tests")
    print("=" * 60)

    # Create test system
    print("\nCreating test system...")
    test_data = create_test_system()
    print(f"  K-points: {len(test_data['kpoints'])}")
    print(f"  Orbitals: {len(test_data['basis_atom_map'])}")
    print(f"  R-vectors: {len(test_data['real_space_matrices'])}")

    # Run validation
    passed = validate_mmn_implementation(
        test_data['kpoints'],
        test_data['eigenvectors'],
        test_data['real_space_matrices'],
        test_data['lattice_vectors'],
        test_data['atom_positions'],
        test_data['basis_atom_map'],
        compute_mmn_matrix
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
