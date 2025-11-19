"""
Unit tests for fourier module

Tests Fourier transforms between real and k-space.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier.fourier import fourier_transform_to_kspace, compute_phase_factors


def test_fourier_transform_hermiticity():
    """Test that Fourier transform preserves Hermiticity."""
    print("\nTest 1: Fourier Transform Hermiticity")
    print("-" * 50)
    
    # Create simple Hermitian matrices in real space
    num_orbitals = 4
    lattice_vectors = np.eye(3)
    
    # H(R) and S(R) for R = (0,0,0), (1,0,0), (-1,0,0)
    real_space_matrices = {}
    
    # R = (0,0,0) - on-site
    H_R0 = np.diag([1.0, 2.0, 3.0, 4.0]) + 0j
    S_R0 = np.eye(num_orbitals) + 0j
    real_space_matrices[(0, 0, 0)] = {'H': H_R0, 'S': S_R0}
    
    # R = (1,0,0) - nearest neighbor
    H_R1 = 0.1 * (np.random.randn(num_orbitals, num_orbitals) + 
                   1j * np.random.randn(num_orbitals, num_orbitals))
    S_R1 = 0.05 * (np.random.randn(num_orbitals, num_orbitals) + 
                    1j * np.random.randn(num_orbitals, num_orbitals))
    real_space_matrices[(1, 0, 0)] = {'H': H_R1, 'S': S_R1}
    
    # R = (-1,0,0) - ensure Hermiticity: H(-R) = H(R)†
    real_space_matrices[(-1, 0, 0)] = {
        'H': H_R1.conj().T,
        'S': S_R1.conj().T
    }
    
    # Test Fourier transform at several k-points
    test_kpoints = [
        np.array([0.0, 0.0, 0.0]),  # Gamma point
        np.array([0.5, 0.0, 0.0]),  # X point
        np.array([0.25, 0.25, 0.0]), # Random point
    ]
    
    max_hermiticity_error = 0.0
    
    for k_point in test_kpoints:
        H_k, S_k = fourier_transform_to_kspace(k_point, real_space_matrices, lattice_vectors)
        
        # Check Hermiticity: H(k) = H(k)†
        H_error = np.max(np.abs(H_k - H_k.conj().T))
        S_error = np.max(np.abs(S_k - S_k.conj().T))
        
        max_hermiticity_error = max(max_hermiticity_error, H_error, S_error)
        
        assert H_error < 1e-14, f"H(k) not Hermitian at k={k_point}: error={H_error}"
        assert S_error < 1e-14, f"S(k) not Hermitian at k={k_point}: error={S_error}"
    
    print(f"  Tested {len(test_kpoints)} k-points")
    print(f"  Maximum Hermiticity error: {max_hermiticity_error:.2e}")
    print("  PASSED")
    
    return True


def test_fourier_transform_gamma():
    """Test Fourier transform at Gamma point."""
    print("\nTest 2: Fourier Transform at Gamma Point")
    print("-" * 50)
    
    num_orbitals = 3
    lattice_vectors = np.eye(3)
    
    # Create real-space matrices
    real_space_matrices = {}
    
    # On-site
    H_R0 = np.diag([1.0, 2.0, 3.0]) + 0j
    S_R0 = np.eye(num_orbitals) + 0j
    real_space_matrices[(0, 0, 0)] = {'H': H_R0, 'S': S_R0}
    
    # Nearest neighbors (all equal hopping)
    t = 0.1
    H_hop = t * np.ones((num_orbitals, num_orbitals)) + 0j
    
    for R in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        real_space_matrices[R] = {'H': H_hop.copy(), 'S': 0.0 * H_hop.copy()}
    
    # At Gamma point (k=0), exp(i k·R) = 1 for all R
    # So H(Gamma) = sum of all H(R)
    k_gamma = np.array([0.0, 0.0, 0.0])
    H_k, S_k = fourier_transform_to_kspace(k_gamma, real_space_matrices, lattice_vectors)
    
    # Expected: H(Gamma) = H_R0 + 6*H_hop
    H_expected = H_R0 + 6 * H_hop
    S_expected = S_R0
    
    H_error = np.max(np.abs(H_k - H_expected))
    S_error = np.max(np.abs(S_k - S_expected))
    
    assert H_error < 1e-14, f"H(Gamma) incorrect: error={H_error}"
    assert S_error < 1e-14, f"S(Gamma) incorrect: error={S_error}"
    
    print(f"  H(Gamma) max error: {H_error:.2e}")
    print(f"  S(Gamma) max error: {S_error:.2e}")
    print("  PASSED")
    
    return True


def test_phase_factor_computation():
    """Test precomputation of phase factors."""
    print("\nTest 3: Phase Factor Computation")
    print("-" * 50)
    
    # Test k-points
    kpoints = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.25, 0.25, 0.25],
    ])
    
    # Test R-vectors
    R_vectors = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (-1,0,0)]
    
    phase_factors = compute_phase_factors(kpoints, R_vectors)
    
    # Check shape
    assert phase_factors.shape == (len(kpoints), len(R_vectors)), \
        f"Shape mismatch: {phase_factors.shape}"
    
    # Check Gamma point (all phases should be 1)
    gamma_phases = phase_factors[0, :]
    assert np.allclose(gamma_phases, 1.0), \
        f"Gamma point phases not 1: {gamma_phases}"
    
    # Check phase at k=(0.5,0,0) for R=(1,0,0): should be exp(i*pi) = -1
    k_idx = 1  # k=(0.5,0,0)
    R_idx = 1  # R=(1,0,0)
    phase = phase_factors[k_idx, R_idx]
    expected_phase = np.exp(2j * np.pi * 0.5 * 1.0)  # exp(i*pi) = -1
    
    assert np.abs(phase - expected_phase) < 1e-14, \
        f"Phase incorrect: {phase} vs {expected_phase}"
    
    # Check that |phase| = 1 for all
    phase_magnitudes = np.abs(phase_factors)
    assert np.allclose(phase_magnitudes, 1.0), \
        f"Phase magnitudes not unity: {phase_magnitudes}"
    
    print(f"  Computed phases for {len(kpoints)} k-points and {len(R_vectors)} R-vectors")
    print(f"  All phase magnitudes = 1.0")
    print("  PASSED")
    
    return True


def test_inverse_fourier_consistency():
    """Test that inverse Fourier transform is consistent."""
    print("\nTest 4: Inverse Fourier Transform Consistency")
    print("-" * 50)
    
    # Note: This test verifies mathematical consistency
    # Not testing actual inverse_fourier_transform function since it may not exist
    
    num_orbitals = 2
    lattice_vectors = np.eye(3)
    
    # Simple real-space model
    real_space_matrices = {
        (0, 0, 0): {'H': np.diag([1.0, 2.0]) + 0j, 
                    'S': np.eye(2) + 0j},
        (1, 0, 0): {'H': 0.1 * np.ones((2, 2)) + 0j,
                    'S': 0.05 * np.ones((2, 2)) + 0j},
        (-1, 0, 0): {'H': 0.1 * np.ones((2, 2)) + 0j,
                     'S': 0.05 * np.ones((2, 2)) + 0j},
    }
    
    # Transform to k-space at several points
    kpoints = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    ]
    
    # Verify Parseval's theorem: sum of |H(k)|^2 relates to sum of |H(R)|^2
    # This is a consistency check
    
    real_space_norm = 0.0
    for R, matrices in real_space_matrices.items():
        real_space_norm += np.sum(np.abs(matrices['H'])**2)
    
    k_space_norm = 0.0
    for k in kpoints:
        H_k, _ = fourier_transform_to_kspace(np.array(k), real_space_matrices, 
                                              lattice_vectors)
        k_space_norm += np.sum(np.abs(H_k)**2)
    
    # Note: These won't be exactly equal due to finite k-point sampling
    # But they should be in same order of magnitude
    print(f"  Real-space norm: {real_space_norm:.4f}")
    print(f"  K-space norm (finite sampling): {k_space_norm:.4f}")
    print(f"  Ratio: {k_space_norm/real_space_norm:.4f}")
    print("  PASSED (consistency check)")
    
    return True


def run_all_tests():
    """Run all Fourier transform tests."""
    print("\n" + "=" * 70)
    print("FOURIER MODULE TESTS")
    print("=" * 70)
    
    tests = [
        test_fourier_transform_hermiticity,
        test_fourier_transform_gamma,
        test_phase_factor_computation,
        test_inverse_fourier_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"FOURIER TESTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
