# Symmetric Midpoint Method - Implementation Analysis

## Summary of Your Changes

You have successfully implemented the **Symmetric Midpoint Method** for computing MMN matrices in the LCAO-to-Wannier90 interface. This replaces my incorrect cross-overlap implementation.

## Changes Made

### 1. New Function: `compute_mmn_matrix()`
**Location:** `lcao_wannier/wannier90.py:14-111`

**Key Algorithm:**
```python
# Step 1: Convert Cartesian b-vector to fractional coordinates
inv_lattice = np.linalg.inv(lattice_vectors.T)
b_frac = np.dot(inv_lattice, b_vector_cart)

# Step 2: Compute midpoint k_mid = k + 0.5 * b_frac
k_mid = k_curr + 0.5 * b_frac

# Step 3: Get S(k_mid) via Fourier transform
_, S_mid = fourier_transform_to_kspace(k_mid, real_space_matrices, lattice_vectors)

# Step 4: Apply symmetric Berry phase correction
# exp[-i * b · (τ_i + τ_j) / 2]
taus = atom_positions[basis_atom_map]
tau_mid = (taus[:, None, :] + taus[None, :, :]) / 2.0
phase_exponent = np.sum(tau_mid * b_vector_cart, axis=2)
phase_matrix = np.exp(-1j * phase_exponent)
S_cross = S_mid * phase_matrix

# Step 5: Project onto eigenvectors
M_mn = C_k.conj().T @ S_cross @ C_next
```

**Physics:**
- Uses **physical b-vector** to compute midpoint (handles periodic boundary conditions correctly)
- Evaluates overlap at **k_mid = k + 0.5*b** instead of averaging k-point coordinates
- Applies **symmetric Berry phase** based on average atomic positions: `(τ_i + τ_j)/2`
- Element-wise phase correction for each orbital pair

### 2. Updated: `write_mmn_file_lcao()`
**Location:** `lcao_wannier/wannier90.py:294-379`

**Changes:**
- Now calls `compute_mmn_matrix()` for each k-point neighbor pair
- Removed my old asymmetric phase correction logic (which treated rows asymmetrically)
- Updated header to indicate "Symmetric Midpoint Method"
- Signature remains compatible with existing code

### 3. Deleted: `compute_cross_overlap()`
**Location:** `lcao_wannier/fourier.py` (removed)

**Rationale:**
- My old function used incorrect formula: `S(k1, k2) = Σ_R e^(iπ(k2-k1)·R) S(R)`
- Failed self-consistency test: S(k,k) ≠ S(k)
- Not hermitian: S(k1,k2) ≠ S(k2,k1)†
- Your midpoint method is the correct approach

### 4. Validation Test: `test_symmetric_midpoint.py`
**Tests:**
1. **Self-consistency:** M(k,k) with b=0 → Identity ✓ PASS (deviation 1e-16)
2. **Hermiticity:** M(k1,k2) = M(k2,k1)† ✓ PASS (error 4e-16)
3. **Boundedness:** Singular values are reasonable ✓ PASS

### 5. Updated: `lcao_wannier/__init__.py`
- Exported `compute_mmn_matrix` in public API

## Theoretical Foundation

### Why Symmetric Midpoint?

The symmetric midpoint method addresses the fundamental issue with LCAO-to-Wannier interfaces:

**Problem:** For non-orthogonal basis sets (LCAO), the overlap between Bloch states at different k-points is non-trivial:
```
<ψ_m^k | ψ_n^(k+b)> ≠ δ_mn
```

**Solution:** Approximate the cross-overlap by:
1. Evaluating S at the midpoint k_mid = k + b/2
2. Applying symmetric phase correction for cell-periodic parts

This is superior to:
- **My approach:** Simple k-difference Fourier transform (mathematically incorrect)
- **Old asymmetric approach:** Phase correction only on one side (breaks hermiticity)

### Comparison with My Failed Attempt

| Aspect | My Cross-Overlap | Your Symmetric Midpoint |
|--------|------------------|-------------------------|
| **Midpoint** | Used k-difference: `e^(i(k2-k1)·R)` | Uses b-vector: `k + 0.5*b` |
| **PBC handling** | Incorrect (averaged k-points) | Correct (uses physical b-vector) |
| **Phase correction** | Asymmetric (rows only) | Symmetric (pairwise average) |
| **Self-consistency** | FAIL: S(k,k) ≠ S(k) | PASS: b=0 → Identity |
| **Hermiticity** | FAIL: S ≠ S† | PASS: M = M† |
| **Unitarity** | Unknown (not tested) | To be tested |

## Impact on Bismuth Workflow

### No Changes Required to Basic Workflow

The existing workflow structure remains identical:

```bash
# Stage 1: Create .win file
python lcao_to_wannier90.py --stage 1 --input tests/Bismuth_basis_40.out \
    --seedname bismuth --projections "Bi:p" --window -25.0 10.0

# Wannier90 preprocessing
wannier90.x -pp bismuth

# Stage 2: Create data files (now uses symmetric midpoint!)
python lcao_to_wannier90.py --stage 2 --input tests/Bismuth_basis_40.out \
    --seedname bismuth --window -25.0 10.0

# Wannier90 run
wannier90.x bismuth
```

### What Changed Under the Hood

**Old behavior (my incorrect code):**
```python
# Computed: S(k1, k2) = Σ_R e^(iπ(k2-k1)·R) S(R)
S_cross = compute_cross_overlap(k_point, k_next, ...)
M_kb = C_k.conj().T @ S_cross @ C_next
```

**New behavior (your symmetric midpoint):**
```python
# Computes: S_mid at k + 0.5*b, with symmetric Berry phase
M_kb = compute_mmn_matrix(k_idx, next_k_idx, ...)
```

### Parameters That Still Need Manual Editing

The Stage 1 script creates a .win file, but the following parameters must be **manually added/edited** in the .win file:

```
num_wann = 12              # Number of Wannier functions
dis_froz_min = -4.5        # Frozen window minimum (eV)
dis_froz_max = 2.0         # Frozen window maximum (eV)
```

**Current workflow:**
1. Run Stage 1 → generates .win with basic parameters
2. **Manually edit .win** to add num_wann and frozen window
3. Run `wannier90.x -pp` → generates .nnkp
4. Run Stage 2 → reads num_wann from .win, generates .eig/.amn/.mmn

**Why manual editing?**
- The script doesn't have command-line flags for `--num-wann` or `--frozen-window`
- These parameters depend on the physical system and desired localization

### Recommendation for Automation

Consider adding command-line arguments to Stage 1:

```python
# In lcao_to_wannier90.py argument parser
parser.add_argument('--num-wann', type=int, help='Number of Wannier functions')
parser.add_argument('--frozen-window', nargs=2, type=float, metavar=('MIN', 'MAX'),
                    help='Frozen window in eV (dis_froz_min dis_froz_max)')
```

Then users could run:
```bash
python lcao_to_wannier90.py --stage 1 --input tests/Bismuth_basis_40.out \
    --seedname bismuth --projections "Bi:p" --window -25.0 10.0 \
    --num-wann 12 --frozen-window -4.5 2.0
```

## Testing Status

### Validation Tests ✅
- Self-consistency: PASS (b=0 gives identity)
- Hermiticity: PASS (M = M†)
- Boundedness: PASS (singular values reasonable)

### Bismuth Workflow 🔄
- Stage 1: Not yet tested (process hung during my attempt)
- Stage 2: Currently running with existing bismuth_fixed files
- MMN unitarity: **PENDING** (need to run `diagnose_wannier_matrices.py`)
- Wannier90 Omega_I: **PENDING** (need to run `wannier90.x`)

## Expected Outcomes

### If Symmetric Midpoint is Correct:

1. **MMN Unitarity:** Should dramatically improve
   - Old (my code): deviation ~4.4, determinants 0.0004-1.04
   - Expected: deviation < 0.01, determinants 0.99-1.01

2. **Omega_I:** Should become positive
   - Old problem: Omega_I = -75.83 (unphysical)
   - Expected: Omega_I = 5-50 Ang² (physical)

3. **Wannier Functions:** Should be properly localized
   - Total spread should be reasonable
   - Centers should correspond to atomic/bond positions

## Potential Issues to Watch

### 1. Condition Number Warning
The validation test showed:
```
Singular Values: range [0.2073, 4.5250]
Condition number: 21.8262
-> WARNING (Matrix may be poorly conditioned, but not necessarily wrong)
```

This is **not necessarily a problem** - it just means some overlap matrix elements are small. But monitor if:
- Condition number becomes > 100 (ill-conditioned)
- Determinants deviate significantly from 1.0
- Wannier90 reports convergence issues

### 2. Midpoint Outside First BZ
For large b-vectors, k_mid = k + 0.5*b might fall outside the first Brillouin zone. This should be fine because:
- The Fourier transform `fourier_transform_to_kspace()` works for any k
- Bloch periodic boundary conditions are automatically satisfied
- But verify that results are sensible for systems with small k-grids

### 3. Phase Correction Magnitude
The symmetric Berry phase `exp[-i*b·(τ_i+τ_j)/2]` could be large if:
- Atomic positions are far from origin
- b-vectors are large (small k-grid)

Monitor: If phase angles are >> 2π, consider re-centering atomic positions.

## Next Steps

1. **Complete Bismuth Stage 2** run and check if files are generated
2. **Run MMN diagnostics:**
   ```bash
   python diagnose_wannier_matrices.py bismuth_fixed
   ```
3. **Run Wannier90:**
   ```bash
   wannier90.x bismuth_fixed
   ```
4. **Check Omega_I** in the .wout file for positive value
5. **Compare spreads** with previous results

## Files Modified by Your Implementation

- ✅ `lcao_wannier/wannier90.py` - Added `compute_mmn_matrix()`, updated `write_mmn_file_lcao()`
- ✅ `lcao_wannier/fourier.py` - Removed `compute_cross_overlap()`
- ✅ `lcao_wannier/__init__.py` - Exported new function
- ✅ `test_symmetric_midpoint.py` - Created validation tests
- ❌ `lcao_wannier/engine.py` - **No changes needed** (signature compatible)
- ❌ `lcao_to_wannier90.py` - **No changes needed** (works with new implementation)

## Conclusion

Your symmetric midpoint implementation is **theoretically sound** and **passes all validation tests**. It correctly addresses the fundamental mathematical errors in my cross-overlap approach.

**No workflow changes are required** - the existing Bismuth workflow will automatically use the new symmetric midpoint method when Stage 2 is run.

The only potential improvement is adding command-line arguments for `num_wann` and `frozen_window` to avoid manual .win file editing.

**Next critical test:** Run the full workflow and verify that:
1. MMN matrices are now unitary
2. Wannier90 produces positive Omega_I
3. Wannier functions are properly localized
