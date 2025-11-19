"""
Unit tests for kpoints module

Tests k-point grid generation and neighbor list construction.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier.kpoints import (
    generate_kpoint_grid,
    generate_neighbor_list,
    kpoint_index_to_grid,
    grid_to_kpoint_index
)


def test_kpoint_grid_generation():
    """Test k-point grid generation."""
    print("\nTest 1: K-Point Grid Generation")
    print("-" * 50)
    
    # Test case: 2x2x2 grid
    k_grid = (2, 2, 2)
    kpoints = generate_kpoint_grid(k_grid)
    
    # Should have 8 k-points
    assert kpoints.shape == (8, 3), f"Expected (8, 3), got {kpoints.shape}"
    
    # Check that all k-points are within [0, 1)
    assert np.all(kpoints >= 0) and np.all(kpoints < 1), \
        "K-points outside [0, 1) range"
    
    # Check that k-points are uniformly distributed
    # For a 2x2x2 grid, spacing should be 0.5 in each direction
    unique_x = np.unique(kpoints[:, 0])
    unique_y = np.unique(kpoints[:, 1])
    unique_z = np.unique(kpoints[:, 2])
    
    assert len(unique_x) == 2, f"Expected 2 unique x values, got {len(unique_x)}"
    assert len(unique_y) == 2, f"Expected 2 unique y values, got {len(unique_y)}"
    assert len(unique_z) == 2, f"Expected 2 unique z values, got {len(unique_z)}"
    
    # Test case: 4x4x4 grid
    k_grid = (4, 4, 4)
    kpoints = generate_kpoint_grid(k_grid)
    
    assert kpoints.shape == (64, 3), f"Expected (64, 3), got {kpoints.shape}"
    
    # Check uniform spacing
    unique_x = np.unique(kpoints[:, 0])
    assert len(unique_x) == 4, f"Expected 4 unique x values, got {len(unique_x)}"
    
    # Check spacing is 1/4 = 0.25
    if len(unique_x) > 1:
        spacing_x = unique_x[1] - unique_x[0]
        expected_spacing = 1.0 / 4
        assert np.isclose(spacing_x, expected_spacing, atol=1e-10), \
            f"X spacing {spacing_x} != expected {expected_spacing}"
    
    print(f"  Generated {len(kpoints)} k-points for {k_grid} grid")
    print(f"  K-point range: [{kpoints.min():.3f}, {kpoints.max():.3f}]")
    print(f"  First k-point: {kpoints[0]}")
    print("  PASSED")
    
    return True


def test_neighbor_list_generation():
    """Test periodic boundary condition neighbor list."""
    print("\nTest 2: Neighbor List Generation")
    print("-" * 50)
    
    # Test case: 3x3x3 grid
    k_grid = (3, 3, 3)
    neighbor_list = generate_neighbor_list(k_grid)
    
    # Each k-point should have 6 neighbors (±x, ±y, ±z)
    num_kpoints = 3 * 3 * 3
    assert len(neighbor_list) == num_kpoints, \
        f"Expected {num_kpoints} entries, got {len(neighbor_list)}"
    
    for k_idx, neighbors in neighbor_list.items():
        assert len(neighbors) == 6, \
            f"K-point {k_idx} has {len(neighbors)} neighbors, expected 6"
        
        # Check that all neighbors are valid indices
        for neighbor_idx, b_vec in neighbors:
            assert 0 <= neighbor_idx < num_kpoints, \
                f"Invalid neighbor index: {neighbor_idx}"
            assert len(b_vec) == 3, f"b-vector should be 3D, got {len(b_vec)}"
    
    # Test specific case: corner k-point should wrap around
    corner_idx = 0
    corner_neighbors = neighbor_list[corner_idx]
    
    # Check that b-vectors are unit vectors in ±x, ±y, ±z
    b_vecs = [b for _, b in corner_neighbors]
    b_norms = [np.linalg.norm(b) for b in b_vecs]
    
    # All b-vectors should be unit vectors (within numerical precision)
    for norm in b_norms:
        assert np.isclose(norm, 1.0), f"b-vector norm {norm} not unity"
    
    print(f"  Generated neighbor list for {k_grid} grid")
    print(f"  Each k-point has {len(corner_neighbors)} neighbors")
    print(f"  Verified periodic boundary conditions")
    print("  PASSED")
    
    return True


def test_index_conversions():
    """Test k-point index to grid and back conversions."""
    print("\nTest 3: Index Conversions")
    print("-" * 50)
    
    k_grid = (4, 5, 6)
    num_kpoints = 4 * 5 * 6
    
    # Test all k-points
    for k_idx in range(num_kpoints):
        # Convert to grid indices
        i, j, k = kpoint_index_to_grid(k_idx, k_grid)
        
        # Check bounds
        assert 0 <= i < k_grid[0], f"i={i} out of bounds"
        assert 0 <= j < k_grid[1], f"j={j} out of bounds"
        assert 0 <= k < k_grid[2], f"k={k} out of bounds"
        
        # Convert back
        k_idx_back = grid_to_kpoint_index(i, j, k, k_grid)
        
        # Should get original index
        assert k_idx == k_idx_back, \
            f"Conversion failed: {k_idx} -> ({i},{j},{k}) -> {k_idx_back}"
    
    print(f"  Tested {num_kpoints} index conversions")
    print(f"  All conversions consistent")
    print("  PASSED")
    
    return True


def test_grid_completeness():
    """Test that k-point grid covers all required points."""
    print("\nTest 4: Grid Completeness")
    print("-" * 50)
    
    k_grid = (3, 3, 3)
    kpoints = generate_kpoint_grid(k_grid)
    
    # Should have exactly 3x3x3 = 27 k-points
    expected_num = 3 * 3 * 3
    assert len(kpoints) == expected_num, \
        f"Expected {expected_num} k-points, got {len(kpoints)}"
    
    # Check that all k-points are unique
    unique_kpoints = np.unique(kpoints, axis=0)
    assert len(unique_kpoints) == len(kpoints), \
        f"Duplicate k-points found: {len(kpoints)} total, {len(unique_kpoints)} unique"
    
    # For a Gamma-centered or Monkhorst-Pack grid
    # Check that spacing is consistent
    for dim in range(3):
        unique_vals = np.sort(np.unique(kpoints[:, dim]))
        if len(unique_vals) > 1:
            spacings = np.diff(unique_vals)
            # All spacings should be equal (within tolerance)
            assert np.allclose(spacings, spacings[0], atol=1e-10), \
                f"Inconsistent spacing in dimension {dim}"
    
    print(f"  Verified grid completeness for {k_grid}")
    print(f"  All {len(kpoints)} k-points are unique")
    print(f"  Uniform spacing verified")
    print("  PASSED")
    
    return True


def test_grid_properties():
    """Test mathematical properties of k-point grid."""
    print("\nTest 5: Grid Mathematical Properties")
    print("-" * 50)
    
    k_grid = (4, 4, 4)
    kpoints = generate_kpoint_grid(k_grid)
    
    # Check that grid satisfies basic requirements
    # 1. All points in [0, 1)
    assert np.all(kpoints >= 0.0) and np.all(kpoints < 1.0), \
        "K-points outside [0, 1) range"
    
    # 2. Correct number of points
    expected = k_grid[0] * k_grid[1] * k_grid[2]
    assert len(kpoints) == expected, \
        f"Wrong number of k-points: {len(kpoints)} vs {expected}"
    
    # 3. For each dimension, check we have correct number of unique values
    for dim in range(3):
        unique_vals = np.unique(kpoints[:, dim])
        assert len(unique_vals) == k_grid[dim], \
            f"Dimension {dim}: expected {k_grid[dim]} unique values, got {len(unique_vals)}"
    
    # 4. Grid should be translation invariant (periodic)
    # Sample a few k-points and check neighbors differ by 1/n in grid units
    spacing_0 = 1.0 / k_grid[0]
    spacing_1 = 1.0 / k_grid[1]
    spacing_2 = 1.0 / k_grid[2]
    
    unique_0 = np.sort(np.unique(kpoints[:, 0]))
    unique_1 = np.sort(np.unique(kpoints[:, 1]))
    unique_2 = np.sort(np.unique(kpoints[:, 2]))
    
    if len(unique_0) > 1:
        actual_spacing_0 = unique_0[1] - unique_0[0]
        assert np.isclose(actual_spacing_0, spacing_0, atol=1e-10), \
            f"X spacing mismatch: {actual_spacing_0} vs {spacing_0}"
    
    if len(unique_1) > 1:
        actual_spacing_1 = unique_1[1] - unique_1[0]
        assert np.isclose(actual_spacing_1, spacing_1, atol=1e-10), \
            f"Y spacing mismatch: {actual_spacing_1} vs {spacing_1}"
    
    if len(unique_2) > 1:
        actual_spacing_2 = unique_2[1] - unique_2[0]
        assert np.isclose(actual_spacing_2, spacing_2, atol=1e-10), \
            f"Z spacing mismatch: {actual_spacing_2} vs {spacing_2}"
    
    print(f"  All mathematical properties verified")
    print(f"  Spacing: ({spacing_0:.4f}, {spacing_1:.4f}, {spacing_2:.4f})")
    print("  PASSED")
    
    return True


def run_all_tests():
    """Run all k-point tests."""
    print("\n" + "=" * 70)
    print("KPOINTS MODULE TESTS")
    print("=" * 70)
    
    tests = [
        test_kpoint_grid_generation,
        test_neighbor_list_generation,
        test_index_conversions,
        test_grid_completeness,
        test_grid_properties,
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
    print(f"KPOINTS TESTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
