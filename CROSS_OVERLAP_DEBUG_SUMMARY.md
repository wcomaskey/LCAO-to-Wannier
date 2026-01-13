# Cross-Overlap Implementation Debug Summary

## Problem Statement

Investigating the root cause of negative Omega_I (-75.83) in Wannier90, which indicates unphysical Wannier functions. Diagnostics revealed MMN matrices have severe unitarity violations (deviation 22.6, determinants 0.059-2.1).

## User's Root Cause Analysis

The user provided a detailed mathematical analysis identifying the fundamental issue:

### The Bug in Original Code
```python
# INCORRECT FORMULA (original code)
M_kb = C_k.conj().T @ S_next @ C_next
```

Where `S_next = S(k+b)` is the same-k overlap at k+b.

### Why It's Wrong

C_k and C_next are eigenvector coefficients in **different Hilbert spaces**:
- C_k: coefficients in Bloch basis at k-point k
- C_next: coefficients in Bloch basis at k-point k+b

The formula treats them as if they're in the same basis, which is mathematically incorrect.

### The Correct Formula

For LCAO Bloch functions χ_μ^k(r) = Σ_R e^(ik·R) φ_μ(r - R), we need:

```
M_mn(k,b) = C†_m(k) @ S(k, k+b) @ C_n(k+b)
```

Where S(k, k+b) is the **cross-overlap** between Bloch basis functions at different k-points:

```
S_μν(k, k+b) = <χ_μ^k | χ_ν^(k+b)> = Σ_R e^(i*b·R) S_μν(R)
```

## Implementation Attempts

### Attempt 1: Simple Cross-Overlap (Plan A)

**Implementation:**
```python
def compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, convention='pi'):
    k_diff = k2 - k1
    factor = np.pi if convention == 'pi' else 2 * np.pi

    S_cross = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)

    for R_tuple, matrices in real_space_matrices.items():
        R = np.array(R_tuple)
        phase = np.exp(1j * factor * np.dot(k_diff, R))
        S_cross += phase * matrices['S']

    return S_cross
```

**Formula used:** S(k1, k2) = Σ_R e^(iπ(k2-k1)·R) S(R)

**Updated MMN computation:**
```python
S_cross = compute_cross_overlap(k_point, k_next, real_space_matrices, lattice_vectors, 'pi')
M_kb = C_k.conj().T @ S_cross @ C_next
```

**Files modified:**
1. `lcao_wannier/fourier.py` - Added `compute_cross_overlap()` function
2. `lcao_wannier/wannier90.py` - Changed `write_mmn_file_lcao()` signature and implementation
3. `lcao_wannier/engine.py` - Updated call site to pass kpoints and real_space_matrices

**Test results:**
```
Test case: k1 = [0.0, 0.0667, 0.0], k2 = [0.0, 0.133, 0.0]

Without cross-overlap:
  M†M diagonal range: [0.9438, 1.0858]

With cross-overlap (no phase correction):
  M†M diagonal range: [0.9275, 1.0670]
  det(M) = -0.623 - 0.564j
  Max deviation from identity: 0.0725
```

**Result:** Small improvement but still NOT unitary (should be < 0.001)

**MMN diagnostics after regeneration:**
- Maximum unitarity deviation: 4.134
- Determinants: min=0.000855, max=1.045, mean=0.391
- Still severe unitarity violations

### Attempt 2: Cross-Overlap + Phase Correction (Plan A + B)

**Hypothesis:** Maybe the cross-overlap needs to be combined with the atomic center phase correction that existed in the old code.

**Phase correction formula (from old code):**
```python
phase_factors = np.exp(-1j * b · τ_μ)
S_phase = phase_factors[:, np.newaxis] * S_cross
```

Where τ_μ is the atomic center position for orbital μ.

**Updated implementation:**
```python
# Compute cross-overlap
S_cross = compute_cross_overlap(k_point, k_next, real_space_matrices, lattice_vectors, 'pi')

# Apply atomic center phase correction
relevant_atom_pos = atom_positions[basis_atom_map]
dot_products = np.dot(relevant_atom_pos, b_vec_cart)
phase_factors = np.exp(-1j * dot_products)
S_phase_corrected = phase_factors[:, np.newaxis] * S_cross

# Compute MMN
M_kb = C_k.conj().T @ S_phase_corrected @ C_next
```

**Test results:**
```
b_vec_cart = [0.0, 1.799, 0.0]
Dot products b·τ (first 10): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Phase factors: all equal to 1.0

Without phase correction:
  M†M diagonal range: [0.9275, 1.0670]

With phase correction:
  M†M diagonal range: [0.9275, 1.0670]  # IDENTICAL!
```

**Result:** Phase correction has ZERO effect because b·τ = 0 for all atoms in this test case (atoms lie in xy-plane, b points in y-direction, but atomic y-coordinates are all 0).

**MMN diagnostics after regeneration:**
- Maximum unitarity deviation: 4.408 (slightly WORSE!)
- Determinants: min=0.000360, max=1.045, mean=0.405

## Critical Discovery: Formula is Fundamentally Wrong

### Test 1: Self-Consistency Check

If the cross-overlap formula is correct, then S(k, k) should equal S(k) from standard Fourier transform.

**Test:**
```python
k1 = k2 = [0.0, 0.667, 0.0]

S_standard = fourier_transform_to_kspace(k1, real_space_matrices, lattice_vectors)[1]
S_cross = compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, 'pi')

diff = np.max(np.abs(S_standard - S_cross))
```

**Result:**
```
Max difference: 1.046
⚠ ERROR: Cross-overlap does NOT match same-k overlap!
```

### Test 2: Hermiticity Check

Cross-overlap should satisfy: S(k1, k2) = S(k2, k1)†

**Test:**
```python
k1 = [0.0, 0.667, 0.0]
k2 = [0.0, 0.733, 0.0]

S_12 = compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, 'pi')
S_21 = compute_cross_overlap(k2, k1, real_space_matrices, lattice_vectors, 'pi')

hermiticity_error = np.max(np.abs(S_12 - S_21.conj().T))
```

**Result:**
```
Max |S(k1,k2) - S(k2,k1)†|: 0.168
⚠ WARNING: Cross-overlap is not hermitian!
```

## Mathematical Analysis of the Error

### Attempted Derivation

For LCAO Bloch functions:
```
χ_μ^k(r) = Σ_R e^(ik·R) φ_μ(r - R)
```

The cross-overlap is:
```
<χ_μ^k1 | χ_ν^k2> = ∫ dr [Σ_R e^(-ik1·R) φ_μ*(r-R)] [Σ_R' e^(ik2·R') φ_ν(r-R')]
                  = Σ_R Σ_R' e^(i(k2·R' - k1·R)) ∫ dr φ_μ*(r-R) φ_ν(r-R')
```

Let r'' = r - R and ΔR = R' - R:
```
                  = Σ_R Σ_ΔR e^(ik2·(R+ΔR) - ik1·R) S_μν(ΔR)
                  = Σ_R e^(i(k2-k1)·R) Σ_ΔR e^(ik2·ΔR) S_μν(ΔR)
```

**Problem:** This does NOT simplify to Σ_R e^(i(k2-k1)·R) S(R) as I assumed!

The second sum Σ_ΔR e^(ik2·ΔR) S_μν(ΔR) depends on k2, so it doesn't factor out cleanly.

### Why My Formula Fails

My formula `S(k1, k2) = Σ_R e^(iπ(k2-k1)·R) S(R)` is incorrect because:

1. **Self-consistency:** When k1 = k2, this gives S(0,0) = Σ_R S(R), which is NOT equal to S(k) = Σ_R e^(iπk·R) S(R)

2. **Wrong derivation:** The correct derivation has TWO sums over R vectors with different phase factors, not a single sum

## Verification of Eigenvector Normalization

To rule out other issues, I verified that selected eigenvectors are properly S-orthonormalized:

**Test:**
```python
C_selected = eigenvectors[k_idx][:, band_indices]  # 16 bands
S_k = overlap_matrices[k_idx]
check = C_selected.conj().T @ S_k @ C_selected
```

**Results:**
```
K-point 0:  C†SC diagonal range: [1.0, 1.0], max off-diag: 1e-15
K-point 112: C†SC diagonal range: [1.0, 1.0], max off-diag: 1e-15
K-point 224: C†SC diagonal range: [1.0, 1.0], max off-diag: 1e-15
```

**Conclusion:** Eigenvectors ARE properly S-orthonormalized. The issue is NOT with eigenvector normalization.

## Summary of Findings

### What Works
1. ✅ Real-space S(R) matrices are available and correctly parsed
2. ✅ Eigenvectors are S-orthonormalized (verified to machine precision)
3. ✅ Same-k overlap S(k) computed correctly via Fourier transform
4. ✅ SOC basis doubling handled correctly

### What Doesn't Work
1. ❌ My cross-overlap formula S(k1, k2) = Σ_R e^(iπ(k2-k1)·R) S(R) is mathematically incorrect
2. ❌ It fails self-consistency: S(k,k) ≠ S(k)
3. ❌ It's not hermitian: S(k1,k2) ≠ S(k2,k1)†
4. ❌ MMN matrices remain non-unitary (deviation ~4)

### Root Cause
The formula for computing cross-overlap from real-space matrices is wrong. The correct formula involves a double sum over R-vectors with coupled phase factors, not a simple single sum.

## Open Questions

1. **What is the correct formula** for S(k1, k2) in terms of S(R)?
   - My derivation suggests it's not a simple Fourier transform
   - May need a double sum or different approach

2. **How do other LCAO-to-Wannier codes handle this?**
   - PySCF has an LCAO-to-Wannier interface
   - Siesta has wannier90 interface
   - What formula do they use?

3. **Is the user's formula S(k1, k2) = Σ_R e^(ib·R) S(R) actually correct?**
   - My tests suggest it's not, but maybe I'm implementing it wrong
   - Need to verify the mathematical derivation more carefully

4. **Alternative approach:** Maybe for LCAO we should use a different overlap definition entirely?

## Recommendations

1. **Consult literature/references** on LCAO-to-Wannier90 interfaces to find the correct formula

2. **Test the OLD code** (before my changes) with Wannier90 to confirm if negative Omega_I actually occurs or if that was a different issue

3. **Consider alternative approaches:**
   - Maybe use pseudo-atomic orbitals (PAO) overlap instead?
   - Maybe transform to orthogonal basis first?
   - Maybe the phase correction is more important than I thought?

4. **Derive formula more carefully** with proper treatment of the double sum

## Test Files Created

1. `test_mmn_normalization.py` - Verifies eigenvector S-orthonormality
2. `test_cross_overlap.py` - Tests cross-overlap vs standard formula
3. `test_cross_overlap_with_phase.py` - Tests phase correction effect
4. `test_cross_overlap_special_cases.py` - Tests self-consistency and hermiticity

## Current Code State

The codebase currently has my (incorrect) cross-overlap implementation:
- `lcao_wannier/fourier.py:172-255` - `compute_cross_overlap()` function
- `lcao_wannier/wannier90.py:192-305` - Modified `write_mmn_file_lcao()`
- `lcao_wannier/engine.py:748-760` - Updated call site

These changes should be **REVERTED** or **FIXED** with the correct formula before proceeding.
