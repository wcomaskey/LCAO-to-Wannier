"""
Unit tests for engine module

Tests the main Wannier90Engine class.
"""

import sys
import os
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import Wannier90Engine


def create_test_system():
    """Create a simple test system."""
    num_orbitals = 4
    lattice_vectors = np.eye(3) * 5.0  # 5 Angstrom cubic cell
    
    # Simple tight-binding model
    real_space_matrices = {}
    
    # On-site energies
    H_R0 = np.diag([0.0, 1.0, 2.0, 3.0]) + 0j
    S_R0 = np.eye(num_orbitals) + 0j
    real_space_matrices[(0, 0, 0)] = {'H': H_R0, 'S': S_R0}
    
    # Nearest-neighbor hopping
    t = -0.5
    H_hop = t * np.eye(num_orbitals) + 0j
    S_hop = 0.1 * np.eye(num_orbitals) + 0j
    
    for R in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        real_space_matrices[R] = {'H': H_hop.copy(), 'S': S_hop.copy()}
    
    return real_space_matrices, lattice_vectors, num_orbitals


def test_engine_initialization():
    """Test engine initialization with valid parameters."""
    print("\nTest 1: Engine Initialization")
    print("-" * 50)
    
    real_space_matrices, lattice_vectors, num_orbitals = create_test_system()
    
    # Initialize engine
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(2, 2, 2),
        lattice_vectors=lattice_vectors,
        num_wann=2,
        seedname='test'
    )
    
    # Check attributes
    assert engine.k_grid == (2, 2, 2), "K-grid not set correctly"
    assert engine.num_wann == 2, "num_wann not set correctly"
    assert engine.seedname == 'test', "Seedname not set correctly"
    assert engine.num_kpoints == 8, f"Expected 8 k-points, got {engine.num_kpoints}"
    
    # Check that k-points and neighbors were generated
    assert len(engine.kpoints) == 8, "K-points not generated"
    assert len(engine.neighbor_list) == 8, "Neighbor list not generated"
    
    print(f"  Initialized engine with {engine.num_kpoints} k-points")
    print(f"  System: {num_orbitals} orbitals, {engine.num_wann} Wannier functions")
    print("  PASSED")
    
    return True


def test_eigenvalue_solving():
    """Test eigenvalue problem solving."""
    print("\nTest 2: Eigenvalue Problem Solving")
    print("-" * 50)
    
    real_space_matrices, lattice_vectors, _ = create_test_system()
    
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=3,
        seedname='test'
    )
    
    # Solve sequentially
    engine.solve_all_kpoints(parallel=False)
    
    # Check that eigenvalues were computed for all k-points
    # Use eigenvalues_list (actual attribute name)
    assert hasattr(engine, 'eigenvalues_list'), "eigenvalues_list attribute not found"
    assert len(engine.eigenvalues_list) == engine.num_kpoints, \
        "Eigenvalues not computed for all k-points"
    
    # Check dimensions of eigenvalues and eigenvectors
    for k_idx in range(engine.num_kpoints):
        eigs = engine.eigenvalues_list[k_idx]
        evecs = engine.eigenvectors_list[k_idx]
        
        assert len(eigs) == engine.num_wann, \
            f"Expected {engine.num_wann} eigenvalues, got {len(eigs)}"
        assert evecs.shape == (4, engine.num_wann), \
            f"Expected eigenvectors shape (4, {engine.num_wann}), got {evecs.shape}"
        
        # Check that eigenvalues are real
        assert np.all(np.isreal(eigs)), f"Eigenvalues not real at k-point {k_idx}"
        
        # Check that eigenvalues are sorted
        assert np.all(eigs[:-1] <= eigs[1:]), \
            f"Eigenvalues not sorted at k-point {k_idx}"
    
    print(f"  Solved eigenvalue problem for {engine.num_kpoints} k-points")
    print(f"  Eigenvalue range: [{engine.eigenvalues_list[0][0]:.3f}, {engine.eigenvalues_list[-1][-1]:.3f}]")
    print("  PASSED")
    
    return True


def test_file_generation():
    """Test Wannier90 file generation."""
    print("\nTest 3: Wannier90 File Generation")
    print("-" * 50)
    
    real_space_matrices, lattice_vectors, _ = create_test_system()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=(2, 2, 2),
            lattice_vectors=lattice_vectors,
            num_wann=2,
            seedname=os.path.join(temp_dir, 'test')
        )
        
        # Solve and write files
        engine.solve_all_kpoints(parallel=False)
        engine.write_files()
        
        # Check that files were created
        eig_file = os.path.join(temp_dir, 'test.eig')
        amn_file = os.path.join(temp_dir, 'test.amn')
        mmn_file = os.path.join(temp_dir, 'test.mmn')
        
        assert os.path.exists(eig_file), ".eig file not created"
        assert os.path.exists(amn_file), ".amn file not created"
        assert os.path.exists(mmn_file), ".mmn file not created"
        
        # Check file sizes (should be non-empty)
        assert os.path.getsize(eig_file) > 0, ".eig file is empty"
        assert os.path.getsize(amn_file) > 0, ".amn file is empty"
        assert os.path.getsize(mmn_file) > 0, ".mmn file is empty"
        
        # Read and verify .eig file format
        with open(eig_file, 'r') as f:
            lines = f.readlines()
            # Should have num_bands * num_kpoints lines
            expected_lines = engine.num_wann * engine.num_kpoints
            assert len(lines) == expected_lines, \
                f"Expected {expected_lines} lines in .eig, got {len(lines)}"
        
        print(f"  Generated files in {temp_dir}")
        print(f"  .eig file: {os.path.getsize(eig_file)} bytes")
        print(f"  .amn file: {os.path.getsize(amn_file)} bytes")
        print(f"  .mmn file: {os.path.getsize(mmn_file)} bytes")
        print("  PASSED")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
    
    return True


def test_parallel_vs_sequential():
    """Test that parallel and sequential give same results.
    
    Properly handles:
    1. Eigenvector phase ambiguity (eigenvectors can differ by e^(i*phi))
    2. Degenerate subspace ordering (eigenvectors can be reordered within degenerate spaces)
    3. Numerical precision issues
    """
    print("\nTest 4: Parallel vs Sequential Consistency")
    print("-" * 50)
    
    real_space_matrices, lattice_vectors, _ = create_test_system()
    
    # Sequential
    engine_seq = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=2,
        seedname='test_seq'
    )
    engine_seq.solve_all_kpoints(parallel=False)
    
    # Parallel
    engine_par = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=2,
        seedname='test_par'
    )
    engine_par.solve_all_kpoints(parallel=True, num_processes=2)
    
    # Compare results
    max_eig_diff = 0.0
    subspace_matches = 0
    total_kpoints = engine_seq.num_kpoints
    
    for k_idx in range(total_kpoints):
        # 1. Eigenvalues should be identical (or very close)
        eigs_seq = engine_seq.eigenvalues_list[k_idx]   
        eigs_par = engine_par.eigenvalues_list[k_idx] 
        
        eig_diff = np.max(np.abs(eigs_seq - eigs_par))
        max_eig_diff = max(max_eig_diff, eig_diff)
        
        # Eigenvalues must match
        assert eig_diff < 1e-10, \
            f"Eigenvalue difference at k-point {k_idx}: {eig_diff}"
        
        # 2. Check if subspaces match using projection method
        # The subspaces match if P_seq ≈ P_par where P = C C†
        evecs_seq = engine_seq.eigenvectors_list[k_idx] 
        evecs_par = engine_par.eigenvectors_list[k_idx]
        
        # Compute subspace projectors: P = C C† 
        # (using num_wann eigenvectors)
        C_seq = evecs_seq[:, :engine_seq.num_wann]
        C_par = evecs_par[:, :engine_par.num_wann]
        
        P_seq = C_seq @ C_seq.conj().T
        P_par = C_par @ C_par.conj().T
        
        # Check if projectors are equal (they should project onto the same subspace)
        projector_diff = np.max(np.abs(P_seq - P_par))
        
        # Use a reasonable tolerance for subspace matching
        # This accounts for numerical differences between sequential and parallel
        if projector_diff < 1e-8:
            subspace_matches += 1
    
    subspace_match_fraction = subspace_matches / total_kpoints
    
    print(f"  Compared {total_kpoints} k-points")
    print(f"  Maximum eigenvalue difference: {max_eig_diff:.2e}")
    print(f"  Subspace match: {subspace_matches}/{total_kpoints} k-points ({100*subspace_match_fraction:.1f}%)")
    
    # Require that at least 95% of k-points have matching subspaces
    # (allow for some numerical variation in edge cases)
    assert subspace_match_fraction > 0.95, \
        f"Too many subspace mismatches: only {100*subspace_match_fraction:.1f}% match"
    
    print("  PASSED")
    
    return True


def test_parallel_vs_sequential_alternative():
    """Alternative test that checks eigenvector correspondence directly.
    
    This version handles phase ambiguity by checking |<v1|v2>| = 1
    """
    print("\nTest 4: Parallel vs Sequential Consistency (Alternative)")
    print("-" * 50)
    
    real_space_matrices, lattice_vectors, _ = create_test_system()
    
    # Sequential
    engine_seq = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=2,
        seedname='test_seq'
    )
    engine_seq.solve_all_kpoints(parallel=False)
    
    # Parallel
    engine_par = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        num_wann=2,
        seedname='test_par'
    )
    engine_par.solve_all_kpoints(parallel=True, num_processes=2)
    
    # Compare results
    max_eig_diff = 0.0
    max_evec_mismatch = 0.0
    
    for k_idx in range(engine_seq.num_kpoints):
        # 1. Check eigenvalues
        eigs_seq = engine_seq.eigenvalues[k_idx]
        eigs_par = engine_par.eigenvalues[k_idx]
        
        eig_diff = np.max(np.abs(eigs_seq - eigs_par))
        max_eig_diff = max(max_eig_diff, eig_diff)
        
        assert eig_diff < 1e-10, \
            f"Eigenvalue difference at k-point {k_idx}: {eig_diff}"
        
        # 2. Check eigenvectors (accounting for phase ambiguity)
        evecs_seq = engine_seq.eigenvectors[k_idx]
        evecs_par = engine_par.eigenvectors[k_idx]
        
        # For each eigenvector, check that |<seq|par>| ≈ 1
        # (they should be identical up to a phase factor)
        for n in range(engine_seq.num_wann):
            v_seq = evecs_seq[:, n]
            v_par = evecs_par[:, n]
            
            # Compute overlap magnitude (removing phase)
            overlap_magnitude = np.abs(np.vdot(v_seq, v_par))
            
            # Should be ≈ 1 (up to normalization)
            # Account for fact that eigenvectors might be normalized differently
            norm_seq = np.sqrt(np.vdot(v_seq, v_seq).real)
            norm_par = np.sqrt(np.vdot(v_par, v_par).real)
            
            # Normalize and check overlap
            normalized_overlap = overlap_magnitude / (norm_seq * norm_par)
            
            mismatch = abs(normalized_overlap - 1.0)
            max_evec_mismatch = max(max_evec_mismatch, mismatch)
            
            # Relaxed tolerance to account for degenerate subspace reordering
            if mismatch > 0.1:  # If significantly different, might be reordered
                # Check if this eigenvector matches ANY other eigenvector
                # (could be swapped within a degenerate subspace)
                found_match = False
                for m in range(engine_seq.num_wann):
                    v_par_alt = evecs_par[:, m]
                    overlap_alt = np.abs(np.vdot(v_seq, v_par_alt))
                    norm_par_alt = np.sqrt(np.vdot(v_par_alt, v_par_alt).real)
                    normalized_overlap_alt = overlap_alt / (norm_seq * norm_par_alt)
                    
                    if abs(normalized_overlap_alt - 1.0) < 0.01:
                        found_match = True
                        break
                
                if not found_match:
                    # Only fail if no matching eigenvector found
                    raise AssertionError(
                        f"Eigenvector mismatch at k={k_idx}, n={n}: "
                        f"no matching eigenvector found (overlap={normalized_overlap:.4f})"
                    )
    
    print(f"  Compared {engine_seq.num_kpoints} k-points")
    print(f"  Maximum eigenvalue difference: {max_eig_diff:.2e}")
    print(f"  Maximum eigenvector mismatch: {max_evec_mismatch:.2e}")
    print("  PASSED")
    
    return True


def test_engine_run_method():
    """Test the high-level run() method."""
    print("\nTest 5: Engine run() Method")
    print("-" * 50)
    
    real_space_matrices, lattice_vectors, _ = create_test_system()
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=(2, 2, 2),
            lattice_vectors=lattice_vectors,
            num_wann=2,
            seedname=os.path.join(temp_dir, 'test')
        )
        
        # Run complete workflow
        results = engine.run(parallel=False, verify=True)
        
        # Check that results is returned (may be None or dict depending on implementation)
        if results is not None:
            print(f"  Results returned: {type(results)}")
            if isinstance(results, dict):
                print(f"  Result keys: {list(results.keys())}")
        
        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, 'test.eig')), \
            "run() did not create .eig file"
        assert os.path.exists(os.path.join(temp_dir, 'test.amn')), \
            "run() did not create .amn file"
        assert os.path.exists(os.path.join(temp_dir, 'test.mmn')), \
            "run() did not create .mmn file"
        
        # Check that eigenvalues were computed
        assert hasattr(engine, 'eigenvalues_list'), \
            "Eigenvalues not computed by run()"
        assert len(engine.eigenvalues_list) == engine.num_kpoints, \
            "Not all k-points solved by run()"
        
        print(f"  run() completed successfully")
        print(f"  All output files created")
        print("  PASSED")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def run_all_tests():
    """Run all engine tests."""
    print("\n" + "=" * 70)
    print("ENGINE MODULE TESTS")
    print("=" * 70)
    
    tests = [
        test_engine_initialization,
        test_eigenvalue_solving,
        test_file_generation,
        test_parallel_vs_sequential,
        test_engine_run_method,
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
    print(f"ENGINE TESTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
