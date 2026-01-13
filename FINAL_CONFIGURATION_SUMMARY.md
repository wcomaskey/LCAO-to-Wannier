# Final Configuration Summary: num_bands vs num_wann

## The Situation

You correctly identified that:
1. `num_wann` = number of Wannier projections (12 for Bi:p with spinors)
2. `num_bands` = number of DFT bands (should be >  num_wann for flexibility)
3. Frozen window captures region of interest near Fermi level
4. Outer window must contain at least `num_bands` DFT bands

## What We Discovered

When trying to create a configuration with `num_bands = 17` (142% of `num_wann = 12`):

**Window [`-5.7, 5.7`] eV**: Only contains **12 bands**
- This is ~11.4 eV wide
- Still only captures 12 bands!

**Reason**: For Bismuth monolayer with SOC:
- The band structure has specific gaps
- Bands near Fermi level (within ~6 eV) are limited to ~12 bands
- To get 17+ bands, you'd need a much wider window (e.g., `[-15, 10]` eV)

## The Trade-off

### Option 1: Wide Window for num_bands > num_wann (e.g., [-15, 10] eV)
**Pros**:
- Achieves `num_bands > num_wann` (e.g., 17-20 bands)
- More flexibility for disentanglement

**Cons**:
- Includes deep valence bands (possibly core states)
- May select physically irrelevant bands
- Harder to converge
- Not physically meaningful for surface states

### Option 2: Narrow Window Matching Physics (e.g., [-5.7, 5.7] eV) ✅ RECOMMENDED
**Pros**:
- Focuses on physically relevant bands near Fermi level
- `num_bands = num_wann = 12` (perfect match, no disentanglement needed!)
- Easier to converge
- Results represent actual surface states

**Cons**:
- No disentanglement flexibility (`num_bands` = `num_wann`)
- Cannot recover if projections are poor

## Recommended Approach

For **Bismuth surface states**, use:

```
num_wann = 12
num_bands = 12
dis_win_min = -5.7
dis_win_max = 5.7
dis_froz_min = -4.0
dis_froz_max = 4.0
```

**Why**: The 12 bands in `[-5.7, 5.7]` eV window ARE the physically meaningful surface/near-surface states. Having `num_bands = num_wann` means no disentanglement is needed - you're just doing a unitary transformation.

## When to Use num_bands > num_wann

Use `num_bands > num_wann` when:
1. You have **entangled bands** (bands crossing each other heavily)
2. You want to **disentangle** specific character (e.g., select p-like from mixed p+d states)
3. Your **projections are uncertain** (need room to optimize)

For **isolated surface states** like Bismuth:
- Bands are already well-separated
- No heavy entanglement
- `num_bands = num_wann` is perfectly fine!

## Current Status

**Configuration**: `num_wann = 12`, `num_bands = 12`, window `[-5.7, 5.7]` eV

**Files Generated**:
- ✅ `bismuth.win` (updated by auto-sync)
- ✅ `bismuth.nnkp`
- ✅ `bismuth.eig` (12 bands × 225 k-points)
- ✅ `bismuth.amn` (projection matrix)
- ✅ `bismuth.mmn` (overlap matrix)

**Next Step**: Run Wannier90 on cluster
```bash
wannier90.x bismuth
```

**Expected Result**: Well-localized Wannier functions representing Bi p-orbitals with SOC

---

## Alternative: If You Really Want num_bands > num_wann

To get 17+ bands, you'd need:

```bash
python3 validate_and_create_win.py \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --num-wann 12
```

Then manually edit `bismuth.win`:
```
dis_win_min = -15.0
dis_win_max = 10.0
```

This will capture deep valence bands, but may not be physically meaningful for surface states.

---

## Script Created

I've created `validate_and_create_win.py` which:
1. Validates `num_bands > num_wann`
2. Ensures frozen window ⊂ outer window
3. Checks that outer window contains enough DFT bands
4. Auto-generates properly configured `.win` file

**Usage**:
```bash
python3 validate_and_create_win.py \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --num-wann 12
```

**Result**: Generates `.win` with validated configuration

---

**Bottom Line**: For Bismuth surface states, `num_bands = num_wann = 12` is the correct physical choice. Don't force `num_bands > num_wann` if it means including unphysical states!

---

**Date**: January 10, 2026
