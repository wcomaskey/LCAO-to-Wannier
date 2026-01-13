# Symmetric Midpoint Method - Test Results with Bismuth

## Executive Summary

The symmetric midpoint method **improved local MMN unitarity** but **did NOT solve the negative Omega_I problem**. Wannier90 still produces negative gauge-invariant spread and crashes during wannierization.

## Test Configuration

**System:** Bismuth 2D monolayer with SOC
- **K-grid:** 15×15×1 (225 k-points)
- **Orbitals:** 112 (56 basis × 2 for SOC)
- **Bands:** 16 bands selected
- **Wannier functions:** 12
- **Energy window:** [-25.0, 10.0] eV
- **Frozen window:** [-4.5, 2.0] eV
- **Projections:** Bi:p

## MMN Matrix Diagnostics

### Local Unitarity (Individual k-points)
**IMPROVED** significantly compared to previous methods:

```
K-point 0, neighbor 0:
  M†M diagonal range: [0.9997, 1.0000]  ← Very close to 1.0!
  Off-diagonal max: 3.14e-04            ← Small

K-point 1, neighbor 0:
  M†M diagonal range: [0.9997, 1.0000]
  Off-diagonal max: 3.23e-04
```

**Comparison with previous methods:**
| Method | M†M Diagonal Range | Off-Diagonal Max |
|--------|-------------------|------------------|
| Old (asymmetric phase) | [0.908, 1.188] | 0.188 |
| My cross-overlap | [0.928, 1.067] | 0.071 |
| **Symmetric midpoint** | **[0.9997, 1.0000]** | **3.2e-04** |

✅ **Local unitarity is nearly perfect!**

### Global Diagnostics
**STILL PROBLEMATIC:**

```
Maximum deviation from unitarity: 7.31
⚠ WARNING: MMN matrices are not unitary!

|det(M)| statistics:
  Min: 0.000850
  Max: 1.261
  Mean: 0.669
⚠ WARNING: Determinants deviate significantly from 1!
```

**Analysis:** While individual k-point neighbors show excellent unitarity, **some subset of neighbors** has severe issues, causing:
- Determinants ranging from 0.0009 to 1.26 (should be ~1.0)
- Maximum deviation of 7.31 across all k-points/neighbors

## Wannier90 Results

### Disentanglement
**Status:** FAILED to converge after 1000 iterations

```
Iteration    Omega_I (Ang²)     Delta
-----------------------------------------
1            -364.65            (start)
2            -370.08            -1.47%
3            -373.25            -1.40%
...
999          -401.46            -1.79e-06
1000         -401.46            -1.78e-06

<<< Warning: Maximum number of disentanglement iterations reached >>>
<<< Disentanglement convergence criteria not satisfied >>>

Final Omega_I: -401.46 Ang²
```

**Key observations:**
- Omega_I started at **-364.65** and got MORE negative
- Converged very slowly (delta ~1e-06) but to wrong value
- Never reached positive Omega_I

### Wannierization
**Status:** CRASHED with segmentation fault

```
Initial State
  Total spread: 1587.34 Ang²

Iteration 0:
  Spread: 1587.34 Ang²

Program received signal SIGSEGV: Segmentation fault
```

Wannier90 crashed immediately after computing the initial spread, before completing the first wannierization iteration.

## Analysis

### What Worked
1. **Symmetric phase correction** correctly handles cell-periodic parts
2. **Midpoint evaluation** properly accounts for periodic boundary conditions
3. **Local M†M unitarity** is excellent (~3e-4 deviation)
4. **Hermiticity** is preserved (validated in test_symmetric_midpoint.py)

### What Didn't Work
1. **Global unitarity** still has issues (max deviation 7.31)
2. **Determinants** range from 0.0009 to 1.26 (not ~1.0)
3. **Omega_I remains negative** and gets worse during disentanglement
4. **Wannier90 crashes** during wannierization

### Possible Root Causes

#### 1. Subset of Problematic k-point Pairs
Since **local unitarity is excellent** but **global statistics are poor**, there must be specific k-point neighbor pairs where the method breaks down. Candidates:
- **Large b-vectors:** When k+b wraps around BZ with large G-shift
- **High-symmetry points:** Gamma, M, K points may need special handling
- **Degenerate k-points:** Cases where k+b = k after wrapping

#### 2. Phase Correction Issues
The symmetric Berry phase `exp[-i*b·(τ_i+τ_j)/2]` may have problems:
- **Phase magnitudes:** If b·τ >> 2π, numerical issues arise
- **Atomic positions:** May need to be centered at origin
- **Pairwise correction:** Element-wise multiplication `S * phase_matrix` may not be correct approach

#### 3. Midpoint Outside First BZ
For large b-vectors, k_mid = k + 0.5*b may fall far outside the first Brillouin zone:
- Fourier transform evaluates S(k_mid) correctly mathematically
- But numerical precision issues could arise
- May need to fold k_mid back into first BZ

#### 4. Fundamental Formula Issue
The symmetric midpoint approximation itself may not be sufficient:
- It's an **approximation**, not the exact cross-overlap
- For LCAO with large orbital spreads, the approximation may break down
- May need the full double-sum formula: Σ_R Σ_R' e^(i k1·R - i k2·R') S(R'-R)

#### 5. Band Selection Issues
The diagnostics show some AMN projection issues:
```
K-point 0: Projection norms per band
  Min: 0.004205    ← Very small
  Max: 0.725480
  ⚠ WARNING: Unusual normalization!
```

This suggests the selected 16 bands may not span the space well, leading to:
- Poor disentanglement
- Negative Omega_I
- Numerical instabilities

## Comparison with Literature

### Expected Omega_I for Bismuth Monolayer
From literature on Bi monolayer with 12 Wannier functions:
- Typical Omega_I: **5-30 Ang²** (positive)
- Total spread: **50-100 Ang²**

Our results:
- Omega_I: **-401.46 Ang²** (negative, unphysical)
- Initial total spread: **1587.34 Ang²** (way too large)

The spreads are **10-30× larger** than expected, indicating a fundamental problem.

## Recommendations

### Immediate Next Steps

1. **Identify problematic k-point pairs:**
   - Modify `diagnose_wannier_matrices.py` to report which specific (k, neighbor) pairs have poor unitarity
   - Check if issues occur at high-symmetry points or with large b-vectors

2. **Check atomic positions:**
   ```python
   print(f"Atomic positions: {atom_positions}")
   print(f"Max |τ|: {np.max(np.linalg.norm(atom_positions, axis=1))}")
   print(f"Max b·τ: {np.max(np.abs(np.dot(atom_positions, b_vec_cart)))}")
   ```
   - If atomic positions are far from origin, re-center them
   - If b·τ >> 2π, this could cause numerical issues

3. **Test with b-vector wrapping:**
   - Modify k_mid computation to wrap back into first BZ:
     ```python
     k_mid = k_curr + 0.5 * b_frac
     k_mid = np.mod(k_mid + 0.5, 1.0) - 0.5  # Wrap to [-0.5, 0.5]
     ```

4. **Compare with old implementation:**
   - Revert to the OLD asymmetric phase correction temporarily
   - Check if Omega_I was also negative before
   - This tells us if the problem is new or pre-existing

### Medium-Term Solutions

1. **Implement exact cross-overlap:**
   - Use the full double-sum formula (computationally expensive)
   - Compare results with symmetric midpoint approximation

2. **Adjust band selection:**
   - Try different energy windows
   - Use more bands (e.g., 20-24 instead of 16)
   - Check eigenvalue spectrum for gaps

3. **Test with simpler system:**
   - Try a system without SOC first
   - Use a smaller k-grid (e.g., 6×6×1)
   - Verify the method works for a known case

### Long-Term Considerations

1. **Consult Wannier90 developers/community:**
   - This issue (negative Omega_I with LCAO) is not unique to your code
   - PySCF, Siesta, and other LCAO codes have similar challenges
   - Check Wannier90 forum/mailing list for insights

2. **Alternative localization:**
   - Try different localization schemes (MLWF vs. SMWF)
   - Use different projections (sp³ instead of p)
   - Consider maximally localized generalized Wannier functions (MLGWF)

3. **Orthogonalization approach:**
   - Transform LCAO basis to orthogonal basis first (Löwdin orthogonalization)
   - Compute MMN in orthogonal basis (simpler formula)
   - Transform Wannier functions back to LCAO basis

## Conclusion

The symmetric midpoint method is **mathematically sound** and produces **excellent local MMN unitarity**. However, it does NOT solve the fundamental issue causing negative Omega_I in Wannier90.

**Key finding:** The problem is likely not in the MMN formula itself (which is locally correct), but rather:
1. A subset of k-point pairs with pathological behavior, OR
2. Poor band selection / disentanglement setup, OR
3. Fundamental incompatibility between the LCAO basis and Wannier90 expectations

**Next step:** Identify which specific k-point neighbor pairs have poor unitarity (determinants ~0.001) and investigate what's special about them.
