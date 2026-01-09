# Wannier Function Localization Analysis

## Current Status

**Spreads after 500 iterations:**
- Individual WF spreads: 40-90 Ang²
- Total spread: 718.39 Ang²
- Omega_I (gauge-invariant): 3.57 Ang²
- Omega_D (diagonal): 154.89 Ang²
- Omega_OD (off-diagonal): 559.94 Ang² ← **LARGE!**

**Key observation:** Omega_OD is 78% of total spread, indicating poor localization.

## Root Cause Analysis

Despite starting from LCAO basis, the selected bands (40-49) are **not** single-orbital character:
- Bands in the frozen window span -8.03 to -1.11 eV
- This is ~7 eV range - likely multiple orbital types mixed
- Bismuth 2D has p-orbitals with different energies that hybridize

## Why LCAO Doesn't Guarantee Localization

1. **Eigenvectors ≠ Atomic Orbitals**
   - C(k) diagonalizes H(k), not minimizes localization
   - Bands mix multiple atomic orbital characters
   - Arbitrary gauge freedom in degenerate/near-degenerate subspaces

2. **Missing Initial Projections**
   - Currently using `projections: random` in .win file
   - Should provide atomic orbital projections as initial guess
   - This guides Wannier90 to physically meaningful gauge

3. **No Disentanglement**
   - 10 bands frozen in [-8.03, -1.11] eV window
   - But bands 50-51 partially cross this range
   - Disentanglement would extract optimal subspace

## Recommended Solutions (in order of priority)

### 1. **Add Atomic Projections** (Highest Impact)

Instead of:
```fortran
begin projections
  random
end projections
```

Use specific atomic orbitals:
```fortran
begin projections
  Bi:p  ! Bismuth p-orbitals
end projections
```

Or more detailed:
```fortran
begin projections
  f=0.0,0.0,0.5:px   ! Atom 1, px orbital
  f=0.0,0.0,0.5:py
  f=0.0,0.0,0.5:pz
  f=0.0,0.0,-0.5:px  ! Atom 2
  f=0.0,0.0,-0.5:py
  f=0.0,0.0,-0.5:pz
  ! Add 4 more projections to reach 10 WFs
end projections
```

**Note:** With `amn_file` from LCAO, projections should match the orbital selection used in A=S@C.

### 2. **Enable Disentanglement** (If bands 50-51 needed)

Add to .win:
```fortran
dis_win_min = -8.5  ! Outer window min (eV, relative to Fermi)
dis_win_max = -0.5  ! Outer window max (include bands 50-51)
dis_froz_min = -8.0 ! Inner frozen window
dis_froz_max = -1.1
dis_num_iter = 1000
dis_mix_ratio = 0.5
```

### 3. **Increase Iterations** (Easy fix)

Change:
```fortran
num_iter = 500
```

To:
```fortran
num_iter = 5000  ! or 10000
```

**But:** More iterations won't fix a fundamental gauge problem. If Omega_OD isn't decreasing after 500 iterations, likely stuck in local minimum.

### 4. **Better Convergence Settings**

```fortran
conv_tol = 1.0e-10
conv_window = 5     ! Increase from 3
num_iter = 5000
```

### 5. **Add Atomic Positions to .win** (Helpful for analysis)

```fortran
begin atoms_frac
Bi  0.000000  0.000000  0.510000
Bi  0.000000  0.000000  0.490000
end atoms_frac
```

## Implementation Priority

**Immediate (will likely solve the issue):**
1. Add proper atomic projections based on band character
2. Increase iterations to 5000

**If still not converging:**
3. Enable disentanglement with outer window
4. Adjust conv_window to 5

**For better analysis:**
5. Add atomic positions to .win

## Next Steps

1. Analyze which orbitals contribute to bands 40-49
   - Check CRYSTAL output for orbital character
   - Or run band decomposition

2. Update .win file with proper projections

3. Re-run Wannier90 and check if Omega_OD decreases

4. If projections don't help, enable disentanglement
