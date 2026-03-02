# Performance Optimization Plan for 500–1000+ Orbital Systems

## Context

The LCAO-to-Wannier90 pipeline currently works well for small systems (Bi bilayer: N=112 orbitals, K=225 k-points). As we target medium to large systems (500–1000+ DFT orbitals), the current implementation hits critical bottlenecks in both time and memory. This plan addresses them in priority order.

### Measured Benchmarks (Apple M1, single-threaded OpenBLAS)

| N | eigh(H,S) | S^{1/2} | matmul | mem (K=225) |
|---|---|---|---|---|
| 112 | 5.6 s | 0.6 s | 0.15 s | 0.18 GB |
| 500 | 82 s | 15 s | 0.34 s | 3.6 GB |

Projected Stage 1 wall times (eigensolve × K=225):
- N=112: ~21 min (current Bi bilayer)
- N=500: ~5 hours
- N=1000: ~40+ hours (infeasible)

### Key Finding: BLAS Configuration
numpy is linked to **single-threaded OpenBLAS** (`USE_OPENMP=0`) on Apple M1. This is the single biggest performance issue — all LAPACK/BLAS calls are using 1 core out of 8.

---

## Phase 0: BLAS Configuration Fix (5-8× speedup, zero code changes)

Switch numpy/scipy to use Apple's Accelerate framework (optimized for M-series, multithreaded):
```bash
pip install --force-reinstall numpy scipy
# pip wheels on macOS ARM link to Accelerate by default
# Alternatively:
conda install "libblas=*=*accelerate"
```

Verify with `python -c "import numpy; numpy.show_config()"` — should show `accelerate` not `openblas`.

**Expected impact**: eigh N=500 drops from 82s → ~10-15s. Stage 1 for N=500 becomes ~45 min instead of 5 hours.

---

## Phase 1: Memory Optimization

### Problem
Engine stores 4 × N×N complex128 matrices per k-point:
- `eigenvalues_list`: K × N (small)
- `eigenvectors_list`: K × N × B (needed)
- `H_k_list`: K × N × N ← **not needed after eigensolve**
- `S_k_list`: K × N × N ← **only needed by SCDM, can recompute**

At N=500, K=225: 3.6 GB total, but 1.8 GB is wasted on H_k/S_k storage.

### Changes

**`lcao_wannier/solver.py`**
- `solve_all_kpoints_sequential()`: Stop returning H_k_list and S_k_list
- `solve_all_kpoints_parallel()`: Same — return only eigenvalues + eigenvectors

**`lcao_wannier/engine.py`**
- Remove `self.H_k_list` and `self.S_k_list` attributes
- Add `self.compute_S_k(k_idx)` method that recomputes S(k) from `self.real_space_matrices` via Fourier transform — O(R × N²), negligible vs O(N³)
- Update all consumers of S_k_list to call `compute_S_k()` on demand
- For SCDM (which needs S_k at every k-point in a loop), the recomputation is R=15 matrix adds vs storing K=225 matrices — clearly better to recompute

**`lcao_wannier/band_selection.py`**
- Modify `scdm_select_projections()` signature: replace `S_k_list` parameter with a callable `get_S_k(k_idx)` that either returns from a list or recomputes

**Memory savings**: ~50% reduction. N=500 goes from 3.6 GB → ~1.8 GB.

---

## Phase 2: Parallelization Improvements

### 2a. Fix Shared Memory in K-point Parallelism

**Problem**: `multiprocessing.Pool` with `partial()` copies `real_space_matrices` dict into each worker via pickle. For N=500, that's ~240 MB per worker.

**`lcao_wannier/solver.py`**
- Use `multiprocessing.shared_memory.SharedMemory` to store R-space matrices as flat arrays in shared memory
- Workers reference the shared data (read-only) instead of copying
- Cleanup shared memory after solve completes

### 2b. Parallelize SCDM K-loop

**`lcao_wannier/band_selection.py`**
- The SCDM loop over k-points (`for ik in range(num_kpoints)`) computes S^{1/2}, P_k, and accumulates into P_L
- Each k-point's contribution to P_L is independent (P_L += ...)
- Split k-points across workers, each produces partial P_L, reduce at end
- QR pivoting runs once on the final P_L (negligible cost)

### 2c. Parallelize MMN Computation

**`lcao_wannier/wannier90.py`**
- MMN has K × 6 ≈ 1350 independent (k, neighbor) pairs
- Each pair: recompute S(k_mid) via Fourier, then M = C†SC
- Use `multiprocessing.Pool.starmap` over the pair list

**Expected speedup**: 4-6× on 8-core M1 (limited by memory bandwidth)

---

## Phase 3: Algorithmic Improvements

### 3a. Vectorized Fourier Assembly (2-3× for Fourier step)

**`lcao_wannier/fourier.py`**
- Current: Python `for` loop over R-vectors
  ```python
  for R_tuple, matrices in real_space_matrices.items():
      phase = exp(2πi k·R)
      H_k += phase * matrices['H']
  ```
- Proposed: Stack into 3D array, use single `np.tensordot` or `np.einsum`
  ```python
  H_stack = np.array([m['H'] for m in real_space_matrices.values()])  # (R, N, N)
  phases = np.exp(2j * np.pi * R_vectors @ k)  # (R,)
  H_k = np.tensordot(phases, H_stack, axes=([0], [0]))  # (N, N)
  ```
- Pre-stack R-space matrices once during engine init (amortized cost)

### 3b. Partial Eigensolve via LOBPCG (50× for B << N)

**`lcao_wannier/solver.py`**
- When `target_num_bands < 0.3 * num_orbitals`, use `scipy.sparse.linalg.lobpcg` instead of full `eigh`
- LOBPCG computes only B eigenvalues/eigenvectors in O(N² × B × iter) vs O(N³)
- For B=20, N=1000: ~50× faster
- Requires initial guess (random or propagated from previous k-point)
- Add `--solver full|iterative` CLI flag (default: auto-select based on B/N ratio)

**Caveats**:
- LOBPCG convergence depends on conditioning of S
- Need to verify eigenvalues match full eigh to within 1e-10
- Only useful when B << N (e.g., projectability selects 20 bands from 1000)

---

## Phase 4: Compiled Backend Assessment

### Verdict: Not recommended at this stage

**Why:** The core operations (eigh, QR, matmul) already call LAPACK/BLAS. A Fortran/C++ wrapper calls the same routines. Python overhead is <1% for N>200 matrices.

| Operation | Python overhead | Compiled would give | Worth it? |
|---|---|---|---|
| `scipy.linalg.eigh` | <1% | 0% (same LAPACK) | No |
| `numpy matmul` | <1% | 0% (same BLAS) | No |
| Fourier loop (15 iters) | ~30% | 2-3× | **numpy vectorization sufficient** |
| SCDM k-loop (225 iters) | ~5% | <2× | Parallelize instead |
| MMN (1350 pairs) | ~10% | <2× | Parallelize instead |

**When compiled code becomes worth it:**
- N > 2000: GPU offloading (cuSolver) for eigensolves
- Sparse matrix exploitation for very large systems
- At that point, link to existing libraries (ELPA, ScaLAPACK) rather than writing new Fortran

### Expected Performance After All Phases

| System | Current | +Phase 0 | +Phase 1+2 | +Phase 3b |
|---|---|---|---|---|
| N=112, K=225 | ~20 min | ~3 min | ~1 min | ~1 min |
| N=500, K=225 | ~5 hr | ~45 min | ~8 min | ~3 min |
| N=1000, K=225 | ~40 hr | ~6 hr | ~1 hr | ~10 min |
| N=2000, K=225 | infeasible | ~50 hr | ~8 hr | ~1.5 hr |

---

## Implementation Order

| Phase | Effort | Impact | Priority |
|---|---|---|---|
| 0: BLAS config | 5 min | 5-8× | **Do now** |
| 1: Memory optimization | 1 day | Enables N>500 | **High** |
| 2: Parallelization | 2 days | 4-6× | **High** |
| 3a: Vectorized Fourier | 2 hours | 2-3× on Fourier | Medium |
| 3b: Partial eigensolve | 1 day | 50× for B<<N | Medium |
| 4: Compiled backend | 1-2 weeks | <2× additional | Not recommended |

---

## Verification

1. **Phase 0**: `numpy.show_config()` shows Accelerate; re-run Bi bilayer Stage 1, confirm ~5× faster
2. **Phase 1**: Compare peak memory (`tracemalloc`) before/after; verify ~50% reduction
3. **Phase 2**: Compare wall time with 1 vs 8 workers; verify ~4-6× speedup
4. **Phase 3b**: Compare eigenvalues from full eigh vs LOBPCG; must agree to 1e-10
5. **All phases**: 206/206 tests pass; Bi bilayer pipeline produces identical output

## Key Files

| File | Phase | Changes |
|---|---|---|
| `lcao_wannier/solver.py` | 1, 2a, 3b | Drop H_k/S_k return; shared memory; LOBPCG |
| `lcao_wannier/engine.py` | 1 | Lazy S_k/H_k recompute; remove stored lists |
| `lcao_wannier/band_selection.py` | 1, 2b | Accept S_k callable; parallelize SCDM |
| `lcao_wannier/wannier90.py` | 2c | Parallelize MMN pairs |
| `lcao_wannier/fourier.py` | 3a | Vectorize R-vector loop with tensordot |
