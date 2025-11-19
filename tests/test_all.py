"""
Comprehensive Test Suite for LCAO-Wannier Package

Run with: python -m pytest tests/ or python tests/test_all.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import (
    Wannier90Engine,
    generate_kpoint_grid,
    generate_neighbor_list,
    fourier_transform_to_kspace,
    prepare_real_space_matrices
)


def create_synthetic_test_data(num_orbitals=4, num_R_vectors=7):
    """Create synthetic H(R) and S(R) matrices for testing."""
    # Define simple cubic lattice vectors
    lattice_constant = 5.0  # Angstroms
    lattice_vectors = np.eye(3) * lattice_constant
    
    # Define lattice vectors to include (origin + 6 nearest neighbors)
    R_vectors = [
        (0, 0, 0),   # On-site
        (1, 0, 0), (-1, 0, 0),  # ±x
        (0, 1, 0), (0, -1, 0),  # ±y
        (0, 0, 1), (0, 0, -1),  # ±z
    ]
    
    real_space_matrices = {}
    
    for R in R_vectors:
        # Create Hamiltonian matrix
        H = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        
        if R == (0, 0, 0):
            # On-site energies (diagonal)
            for i in range(num_orbitals):
                H[i, i] = -2.0 + 0.1 * i
        else:
            # Hopping terms
            t = -0.5
            for i in range(num_orbitals - 1):
                H[i, i+1] = t * (1 + 0.05j)
                H[i+1, i] = t * (1 - 0.05j)
        
        # Create overlap matrix
        S = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        
        if R == (0, 0, 0):
            S = np.eye(num_orbitals, dtype=np.complex128)
        else:
            overlap = 0.1
            for i in range(num_orbitals):
                S[i, i] = overlap
        
        real_space_matrices[R] = {'H': H, 'S': S}
    
    return real_space_matrices, lattice_vectors


def test_kpoint_grid_generation():
    """Test k-point grid generation."""
    print("\nTest 1: K-Point Grid Generation")
    print("-" * 70)
    
    k_grid = (2, 3, 4)
    kpoints = generate_kpoint_grid(k_grid)
    
    expected_num = k_grid[0] * k_grid[1] * k_grid[2]
    
    assert len(kpoints) == expected_num, "K-point count mismatch!"
    assert kpoints.shape == (expected_num, 3), "K-point shape mismatch!"
    assert np.all(kpoints >= 0) and np.all(kpoints < 1), "K-points out of range!"
    
    print(f"  K-grid: {k_grid}")
    print(f"  Generated {len(kpoints)} k-points")
    print("  ✓ PASSED")
    return True


def test_neighbor_list_generation():
    """Test neighbor list generation."""
    print("\nTest 2: Neighbor List Generation")
    print("-" * 70)
    
    k_grid = (3, 3, 3)
    neighbor_list = generate_neighbor_list(k_grid)
    
    num_kpoints = k_grid[0] * k_grid[1] * k_grid[2]
    
    assert len(neighbor_list) == num_kpoints, "Wrong number of k-points in neighbor list!"
    
    for k_idx in range(num_kpoints):
        assert len(neighbor_list[k_idx]) == 6, f"K-point {k_idx} has wrong number of neighbors!"
    
    print(f"  K-grid: {k_grid}")
    print(f"  {num_kpoints} k-points, each with 6 neighbors")
    print("  ✓ PASSED")
    return True


def test_fourier_transform():
    """Test Fourier transform."""
    print("\nTest 3: Fourier Transform")
    print("-" * 70)
    
    real_space_matrices, lattice_vectors = create_synthetic_test_data(num_orbitals=4)
    k_point = np.array([0.25, 0.25, 0.25])
    
    H_k, S_k = fourier_transform_to_kspace(k_point, real_space_matrices, lattice_vectors)
    
    # Check Hermiticity
    H_hermitian = np.allclose(H_k, H_k.conj().T)
    S_hermitian = np.allclose(S_k, S_k.conj().T)
    
    assert H_hermitian, "H(k) is not Hermitian!"
    assert S_hermitian, "S(k) is not Hermitian!"
    
    print(f"  Test k-point: {k_point}")
    print(f"  H(k) is Hermitian: {H_hermitian}")
    print(f"  S(k) is Hermitian: {S_hermitian}")
    print("  ✓ PASSED")
    return True


def test_engine_initialization():
    """Test Wannier90Engine initialization."""
    print("\nTest 4: Engine Initialization")
    print("-" * 70)
    
    real_space_matrices, lattice_vectors = create_synthetic_test_data(num_orbitals=6)
    
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(2, 2, 2),
        lattice_vectors=lattice_vectors,
        num_wann=4,
        seedname="test"
    )
    
    assert engine.num_kpoints == 8, "Wrong number of k-points!"
    assert engine.num_orbitals == 6, "Wrong number of orbitals!"
    assert engine.num_wann == 4, "Wrong number of Wannier functions!"
    
    print(f"  Number of k-points: {engine.num_kpoints}")
    print(f"  Number of orbitals: {engine.num_orbitals}")
    print(f"  Number of Wannier functions: {engine.num_wann}")
    print("  ✓ PASSED")
    return True


def test_complete_workflow():
    """Test complete engine workflow."""
    print("\nTest 5: Complete Workflow")
    print("-" * 70)
    
    real_space_matrices, lattice_vectors = create_synthetic_test_data(num_orbitals=6)
    
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=4,
        seedname="test_complete"
    )
    
    # Run workflow
    results = engine.run(parallel=False, verify=True)
    
    # Check output files exist
    import os
    assert os.path.exists("test_complete.eig"), ".eig file not created!"
    assert os.path.exists("test_complete.amn"), ".amn file not created!"
    assert os.path.exists("test_complete.mmn"), ".mmn file not created!"
    
    # Check verification results
    assert results is not None, "No verification results!"
    assert 'hermiticity' in results, "Missing hermiticity check!"
    assert 'orthonormality' in results, "Missing orthonormality check!"
    
    print(f"  ✓ All files created")
    print(f"  ✓ Verification checks passed")
    print("  ✓ PASSED")
    return True


def test_parallel_consistency():
    """Test that parallel and sequential give same results."""
    print("\nTest 6: Parallel vs Sequential Consistency")
    print("-" * 70)
    
    real_space_matrices, lattice_vectors = create_synthetic_test_data(num_orbitals=6)
    
    # Sequential
    engine_seq = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=4,
        seedname="test_seq"
    )
    engine_seq.solve_all_kpoints(parallel=False)
    
    # Parallel
    engine_par = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=4,
        seedname="test_par"
    )
    engine_par.solve_all_kpoints(parallel=True)
    
    # Compare results
    max_eig_diff = 0.0
    for k_idx in range(len(engine_seq.eigenvalues_list)):
        diff = np.max(np.abs(
            engine_seq.eigenvalues_list[k_idx] - engine_par.eigenvalues_list[k_idx]
        ))
        max_eig_diff = max(max_eig_diff, diff)
    
    assert max_eig_diff < 1e-10, "Parallel and sequential results differ!"
    
    print(f"  Max eigenvalue difference: {max_eig_diff:.2e}")
    print("  ✓ PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS FOR LCAO-WANNIER PACKAGE")
    print("=" * 70)
    
    tests = [
        test_kpoint_grid_generation,
        test_neighbor_list_generation,
        test_fourier_transform,
        test_engine_initialization,
        test_complete_workflow,
        test_parallel_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
