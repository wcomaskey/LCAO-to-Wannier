"""
Basic profiling comparison of Method 1 (projectability) vs Method 2 (direct).

Run with: pytest tests/test_method_profiling.py -v -s
"""

import numpy as np
import time

from lcao_wannier import Wannier90Engine
from lcao_wannier.projectability import (
    compute_band_projectability,
    select_bands_by_projectability,
)


def create_benchmark_system(num_orbitals=20):
    """Create a test system for profiling."""
    lattice_vectors = np.eye(3) * 5.0
    real_space_matrices = {}

    H_R0 = np.diag(np.arange(num_orbitals, dtype=float)) + 0j
    S_R0 = np.eye(num_orbitals) + 0j
    # Add small off-diagonal for realism
    for i in range(num_orbitals - 1):
        H_R0[i, i + 1] = 0.1 + 0j
        H_R0[i + 1, i] = 0.1 + 0j
        S_R0[i, i + 1] = 0.02 + 0j
        S_R0[i + 1, i] = 0.02 + 0j

    real_space_matrices[(0, 0, 0)] = {'H': H_R0, 'S': S_R0}

    for R in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
              (0, 0, 1), (0, 0, -1)]:
        H_hop = -0.3 * np.eye(num_orbitals) + 0j
        S_hop = 0.05 * np.eye(num_orbitals) + 0j
        real_space_matrices[R] = {'H': H_hop, 'S': S_hop}

    return real_space_matrices, lattice_vectors


def test_profile_method1_vs_method2():
    """Profile Method 1 (projectability) vs Method 2 (direct)."""
    num_orbitals = 20
    real_space_matrices, lattice_vectors = create_benchmark_system(num_orbitals)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(3, 3, 3),
        lattice_vectors=lattice_vectors,
        seedname='profile_test',
    )
    engine.solve_all_kpoints(parallel=False)

    # Profile Method 1: projectability computation + selection
    t0 = time.perf_counter()
    result = select_bands_by_projectability(
        engine.eigenvectors_list,
        engine.S_k_list,
        threshold=0.9,
        verbose=False,
    )
    t1 = time.perf_counter()
    method1_time = t1 - t0

    # Profile Method 2: direct setup (just assignments)
    t0 = time.perf_counter()
    engine.num_wann = engine.num_orbitals
    engine.selected_band_indices = np.arange(engine.num_orbitals)
    engine.selected_orbital_indices = np.arange(engine.num_orbitals)
    t1 = time.perf_counter()
    method2_time = t1 - t0

    print(f"\n{'='*60}")
    print(f"PROFILING: Method 1 vs Method 2")
    print(f"{'='*60}")
    print(f"System: {num_orbitals} orbitals, {engine.num_kpoints} k-points")
    print(f"Method 1 (projectability): {method1_time:.6f} s")
    print(f"  Selected {result.num_wann}/{num_orbitals} bands")
    print(f"  Min projectability: {result.band_projectabilities.min():.6f}")
    print(f"  Max projectability: {result.band_projectabilities.max():.6f}")
    print(f"Method 2 (direct setup): {method2_time:.6f} s")
    print(f"{'='*60}")

    # Method 2 setup should be essentially instant
    assert method2_time < 0.1
    # Method 1 should complete in reasonable time
    assert method1_time < 5.0


def test_profile_scaling():
    """Check that projectability computation scales reasonably."""
    times = []
    sizes = [4, 8, 16]

    for n in sizes:
        matrices, lattice = create_benchmark_system(n)
        engine = Wannier90Engine(
            real_space_matrices=matrices,
            k_grid=(2, 2, 2),
            lattice_vectors=lattice,
            seedname='scale_test',
        )
        engine.solve_all_kpoints(parallel=False)

        t0 = time.perf_counter()
        compute_band_projectability(engine.eigenvectors_list, engine.S_k_list)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"\n{'='*60}")
    print(f"SCALING: Projectability computation")
    print(f"{'='*60}")
    for n, t in zip(sizes, times):
        print(f"  {n:3d} orbitals: {t:.6f} s")
    print(f"{'='*60}")

    # All should complete quickly for these small sizes
    for t in times:
        assert t < 5.0
