# Can `num_bands` Be Changed Before Final Wannier90 Run?

## Direct Answer

**NO**. You **CANNOT** change `num_bands` in the `.win` file after creating the `.eig`, `.amn`, and `.mmn` files.

You **MUST** change it **BEFORE** creating those files (i.e., before running Stage 2).

---

## Why?

The `.eig`, `.amn`, and `.mmn` files are **structured data files** where each k-point has exactly `num_bands` entries:

### `.eig` File Structure
```
band  kpt  eigenvalue
1     1    -2.111735
2     1    -2.111734
...
12    1     2.915517    ← num_bands = 12
1     2    -2.123266    ← Next k-point starts
2     2    -2.123266
...
```

If you change `num_bands` from 12 to 16 in `.win` after these files are created:
- Wannier90 expects 16 bands per k-point
- But `.eig` only has 12 bands per k-point
- When Wannier90 tries to read band 13 of k-point 1, it encounters band 1 of k-point 2
- **ERROR: "Wanted band: 13 found band: 1"**

This is exactly the error you encountered initially!

---

## Correct Workflow

### Option 1: Fix `num_bands` in `.win`, Then Regenerate Files

```bash
# 1. Edit bismuth.win manually or let auto-update fix it
#    Set: num_bands = 12 (or whatever value matches your energy window)

# 2. Re-run preprocessing
wannier90.x -pp bismuth

# 3. Re-run Stage 2 (regenerates .eig, .amn, .mmn with correct num_bands)
python3 lcao_to_wannier90.py --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -12.0 6.0

# 4. Run Wannier90
wannier90.x bismuth
```

### Option 2: Use Auto-Update Feature (Already Implemented)

When Stage 2 detects a mismatch between the `.win` file's `num_bands` and the actual number of bands in the energy window, it automatically updates the `.win` file and provides instructions:

```
⚠ IMPORTANT: Updating bismuth.win
  num_bands: 16 → 12
  Reason: Energy window contains 12 bands, not 16
  ✓ Updated bismuth.win

  ============================================================================
  NEXT STEPS:
  1. Re-run preprocessing: wannier90.x -pp bismuth
  2. Re-run Stage 2:       python3 lcao_to_wannier90.py --stage 2 ...
  ============================================================================
```

**You must follow those instructions** - the files need to be regenerated!

---

## What Determines `num_bands`?

`num_bands` is determined by how many bands fall within the energy window you specify:

```
# In bismuth.win:
dis_win_min = -12.0   # eV (relative to Fermi level)
dis_win_max = 6.0     # eV
```

For Bismuth with SOC (112 total bands):
- Energy window `[-12, 6]` eV captures **~16 bands** near Fermi level
- Stage 2 selects the **12 closest to Fermi** (because `num_wann = 12`)
- **Result**: `num_bands = 12` should be set in `.win`

---

## Timeline: When Can You Change What?

| Parameter | When to Change | Effect |
|-----------|---------------|--------|
| `num_wann` | **Before preprocessing** | Changes number of Wannier functions to generate |
| `num_bands` | **Before Stage 2** | Changes how many bands to include in disentanglement |
| Energy windows | **Before Stage 2** | Determines which bands are selected |
| Projections | **Before preprocessing** | Changes initial guess for Wannier functions |

**After Stage 2 completes**:
- The `.eig`, `.amn`, `.mmn` files are **locked** to the current `num_bands`
- The **only** step left is running `wannier90.x` (final calculation)
- Changing `.win` at this point **will cause errors**

---

## Your Specific Situation

### What Happened
1. You ran Stage 2 with default energy window `[-5, 3]` eV
2. This window selected the wrong bands → poor localization (Omega_OD = 526 Ang²)
3. Your `.win` file has better windows: `[-12, 6]` eV
4. **Problem**: The `.eig`/`.amn`/`.mmn` files don't match the `.win` file's intended configuration

### The Fix (Already Done)
I ran `fix_and_rerun.sh` which:
1. ✅ Re-ran preprocessing: `wannier90.x -pp bismuth`
2. ✅ Re-ran Stage 2 with correct window: `--window -12.0 6.0`
3. ✅ Regenerated `.eig`/`.amn`/`.mmn` with correct band selection (bands 40-51)

### Current Status
- **Files regenerated**: bismuth.eig, bismuth.amn, bismuth.mmn (Jan 10 01:18)
- **Band selection**: 12 bands (indices 40-51) from window `[-12, 6]` eV
- **Validation**: ✅ Bands span Fermi level (4 below, 2 crossing, 6 above)
- **Average distance from E_F**: 2.09 eV (good!)

### Next Step
Run Wannier90 final calculation:
```bash
./external/wannier90-3.1.0/wannier90.x bismuth
```

**Expected result**: Much better localization (Omega_OD < 100 Ang²) compared to previous run (Omega_OD = 526 Ang²)

---

## Summary

**Question**: "Can I change `num_bands` before the final Wannier90 run or must I change it before I create the mmn and amn files?"

**Answer**: You **must** change it **before** creating the `.mmn` and `.amn` files. Once those files exist, changing `num_bands` in `.win` will cause a mismatch error. You must regenerate the files by re-running preprocessing and Stage 2.

**Current Status**: ✅ Files successfully regenerated with correct configuration. Ready for final Wannier90 run.

---

## Implementation Details

The auto-update mechanism was added to prevent this confusion:

**Location**: `lcao_wannier/engine.py` lines 679-705

```python
# Validate/update .win file's num_bands to match .eig file
actual_bands_written = len(band_indices)
win_file = f"{self.seedname}.win"

try:
    from .win_file import read_win_parameter, update_win_parameter
    current_num_bands = read_win_parameter(win_file, 'num_bands')

    if current_num_bands is not None and current_num_bands != actual_bands_written:
        if verbose:
            print(f"\n⚠ IMPORTANT: Updating {win_file}")
            print(f"  num_bands: {current_num_bands} → {actual_bands_written}")
            # ... instructions ...

        update_win_parameter(win_file, 'num_bands', actual_bands_written)
except Exception as e:
    if verbose:
        print(f"\n⚠ Warning: Could not validate/update .win file: {e}")
```

This ensures users get clear guidance when a mismatch is detected!

---

**Date**: January 10, 2026
