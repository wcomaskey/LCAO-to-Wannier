# Proper Projections for Spinor Systems

## Critical Understanding: Spinors Double Everything

When `spinors = .true.`, **each projection generates 2 Wannier functions** (spin-up and spin-down components).

Therefore:
- **10 Wannier functions** → need **5 projections**
- **12 Wannier functions** → need **6 projections**
- etc.

## Your Corrected System

**Atomic positions (fractional):**
```
Atom 1 (Bi):  0.333298   0.166649   0.001797
Atom 2 (Bi): -0.333298  -0.166649  -0.001797
```

**Note:** z-coordinates are ~0.002 (not 0.5!) because:
- Actual separation: ±0.90 Å in Cartesian
- Lattice c = 500 Å (large vacuum for 2D)
- Fractional: 0.90/500 = 0.0018 ≈ 0.002

## Recommended Projection Options

### **Option 1: Automatic (Recommended - Easiest)**

```fortran
begin projections
Bi:p
end projections
```

**How it works:**
- Wannier90 finds all Bi atoms from `atoms_frac`
- Places 3 p-orbitals (px, py, pz) on each Bi
- Total: 2 atoms × 3 orbitals = 6 projections
- With spinors: 6 × 2 = **12 WFs**

**Issue:** This gives 12 WFs but you want 10!

**Solutions:**
a) Use `num_wann = 12` and include bands 50-51 (needs disentanglement)
b) Manually specify only 5 projections (see Option 2)

### **Option 2: Explicit - 10 WFs (5 projections)**

You need to decide which 5 orbitals make sense. Common choices:

**A. Three p-orbitals on atom 1, two on atom 2:**
```fortran
begin projections
f=0.333298,0.166649,0.001797:px
f=0.333298,0.166649,0.001797:py
f=0.333298,0.166649,0.001797:pz
f=-0.333298,-0.166649,-0.001797:px
f=-0.333298,-0.166649,-0.001797:py
end projections
```
Total: 5 × 2 (spinor) = **10 WFs** ✓

**B. If bands have mixed s-p character:**
```fortran
begin projections
f=0.333298,0.166649,0.001797:s
f=0.333298,0.166649,0.001797:px
f=0.333298,0.166649,0.001797:py
f=-0.333298,-0.166649,-0.001797:px
f=-0.333298,-0.166649,-0.001797:py
end projections
```

**C. Symmetric - px, py, s on both:**
```fortran
begin projections
f=0.333298,0.166649,0.001797:s
f=0.333298,0.166649,0.001797:px
f=-0.333298,-0.166649,-0.001797:s
f=-0.333298,-0.166649,-0.001797:px
f=0.333298,0.166649,0.001797:py
end projections
```

**Which to choose?**
- **You need to check the band character** from CRYSTAL output
- Look at which orbitals contribute to bands 40-49
- Match projections to dominant orbital types

## With Disentanglement (Recommended)

If bands 50-51 have partial overlap, use disentanglement:

```fortran
num_wann = 10
num_bands = 12  ! Include bands 50-51

! Outer window (include overlapping bands)
dis_win_min = -8.5
dis_win_max = -0.5

! Inner frozen window (your original 10 bands)
dis_froz_min = -8.0
dis_froz_max = -1.1

dis_num_iter = 1000
dis_mix_ratio = 0.5

begin projections
Bi:p  ! This gives 12 projections (6 orbitals × 2 spinor)
end projections
```

**Advantages:**
- Easier projection specification
- Automatically finds optimal 10-dimensional subspace
- Better handles band crossings

## Common Mistakes to Avoid

❌ **WRONG: Doubling projections manually**
```fortran
! Don't do this with spinors!
begin projections
f=0.333,0.167,0.002:px
f=0.333,0.167,0.002:px  ! Duplicate - wrong!
end projections
```
This would give 4 WFs, not 2!

❌ **WRONG: Using z=0.5 for 500 Å cell**
```fortran
f=0.0,0.0,0.5:px  ! This is 250 Å, not 0.9 Å!
```

✓ **CORRECT: Use actual fractional coordinates**
```fortran
f=0.333298,0.166649,0.001797:px  ! Matches real atomic position
```

## Testing Your Projections

After running Wannier90, check:

1. **Initialization:**
   ```
   Number of Wannier Functions: 10
   Number of input projections:  5  (or 10 for automatic Bi:p with correction)
   ```

2. **Initial spread should decrease:**
   ```
   Initial: Omega_OD ~560 Ang²  (your current value)
   After 100 iter: Should drop to <300 Ang²
   Converged: Target <100 Ang² (ideally <50)
   ```

3. **If spreads don't improve:**
   - Projections don't match band character
   - Try different orbital combinations
   - Enable disentanglement

## Summary Recommendations

**Quick test (Option 1):**
Use `bismuth_test_corrected.win` with `Bi:p` - simple but gives 12 WFs

**Production (Option 2):**
1. Check band character from CRYSTAL
2. Manually specify 5 projections matching dominant orbitals
3. Use fractional coords: (±0.333, ±0.167, ±0.002)

**Best approach (Option 3):**
Enable disentanglement, use `Bi:p`, let Wannier90 extract optimal 10-dimensional subspace
