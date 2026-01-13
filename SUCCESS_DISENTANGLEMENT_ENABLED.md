# SUCCESS: Disentanglement Enabled (num_bands > num_wann)

## Problem Solved

The auto-update feature was **automatically changing `num_bands` to match the number of bands in the window**, preventing us from having `num_bands > num_wann` for disentanglement.

## Solution

1. **Disabled auto-update** in `lcao_wannier/engine.py` lines 679-708
2. **Set wide energy window**: `[-15.0, 10.0]` eV
3. **Manual configuration**: `num_wann = 12`, `num_bands = 18`

## Final Configuration

```
# bismuth.win
num_wann = 12
num_bands = 18

dis_win_min = -15.0
dis_win_max = 10.0
dis_froz_min = -6.0
dis_froz_max = 4.0
```

## Results

✅ **Energy window `[-15, 10]` eV** → Contains **22 bands**
✅ **Selected 18 bands** (bands 40-57) → **Closest to Fermi level**
✅ **num_bands = 18 > num_wann = 12** → **Disentanglement ENABLED**
✅ **Band distribution**: 4 below E_F, 2 crossing, 12 above (good!)
✅ **Files generated**:
- `bismuth.eig`: 4050 eigenvalues (18 bands × 225 k-points) ✓
- `bismuth.amn`: 72900 matrix elements ✓
- `bismuth.mmn`: 583200 matrix elements ✓

## What Changed

### Before (Broken)
```
Window: [-5.7, 5.7] eV → 12 bands available
num_bands = 12 (auto-updated from 17!)
num_wann = 12
Result: No disentanglement, negative Omega_I
```

### After (Fixed)
```
Window: [-15, 10] eV → 22 bands available
num_bands = 18 (manually set, auto-update DISABLED)
num_wann = 12
Result: Disentanglement enabled, should get positive Omega_I
```

## Code Changes

**File**: `lcao_wannier/engine.py` lines 679-708

**Change**: Commented out the auto-update feature

```python
# DISABLED: Auto-update feature commented out to allow num_bands > actual bands in window
# This is needed for disentanglement where num_bands should be > num_wann
#
# # Validate/update .win file's num_bands to match .eig file
# ... (27 lines commented out)
```

**Reason**: The auto-update was well-intentioned (prevent mismatch errors) but prevented proper disentanglement configuration.

## Next Steps

Run Wannier90 on cluster with these files:

```bash
wannier90.x bismuth
```

**Expected Results**:
- ✅ **Positive Omega_I** (gauge-invariant spread > 0)
- ✅ **Omega_OD < 100 Ang²** (good localization)
- ✅ **Individual spreads < 10 Ang²** (well-localized WFs)
- ✅ **Convergence** in < 1000 iterations

The disentanglement will:
1. Take the 18 DFT bands (40-57)
2. Find the optimal 12-dimensional subspace with **Bi p-character**
3. Maximize localization within that subspace
4. Produce 12 maximally-localized Wannier functions

## Files to Transfer to Cluster

```bash
scp bismuth.{win,nnkp,eig,amn,mmn} user@cluster:~/bismuth/
```

All files are now properly configured with `num_bands = 18 > num_wann = 12`!

---

**Date**: January 10, 2026
**Status**: ✅ READY FOR WANNIER90
