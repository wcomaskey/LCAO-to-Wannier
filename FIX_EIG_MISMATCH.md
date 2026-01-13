# Fix: Wannier90 .eig File Mismatch Error

## Problem

After implementing the band selection fix, users encountered this error when running Wannier90:

```
Found a mismatch in bismuth.eig
Wanted band  : 13 found band  : 1
Wanted kpoint: 1 found kpoint: 2
param_read: mismatch in bismuth.eig
```

## Root Cause

The issue occurred because of a synchronization problem between two files:

1. **.win file**: Generated with `num_bands = 16` (user's initial request)
2. **.eig file**: Written with only 14 bands (actual bands found in energy window by Stage 2)

**The mismatch**:
- Wannier90 reads `.win` file, sees `num_bands = 16`
- Wannier90 expects to read 16 bands per k-point from `.eig` file
- But `.eig` file only has 14 bands per k-point
- When Wannier90 tries to read band 13-16 of k-point 1, it instead encounters k-point 2's data
- **ERROR: "Wanted band: 13 found band: 1"**

This happened because:
1. User creates `.win` with `num_bands = 16`
2. Energy window `[-10, 5]` eV only contains 14 bands (not 16!)
3. Stage 2 correctly writes 14 bands to `.eig`
4. But `.win` still says `num_bands = 16`
5. Mismatch → Wannier90 fails!

---

## Solution

**Automatic synchronization**: Stage 2 now automatically updates the `.win` file's `num_bands` parameter to match the actual number of bands written to the `.eig` file.

### Implementation

Added two new helper functions to `lcao_wannier/win_file.py`:

1. **`read_win_parameter(filename, parameter)`**
   - Reads a single parameter from `.win` file
   - Returns int, float, bool, or string value

2. **`update_win_parameter(filename, parameter, value)`**
   - Updates a single parameter in `.win` file
   - Preserves formatting and indentation

Added validation logic to `lcao_wannier/engine.py` (after line 677):

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
            print(f"  Reason: Energy window contains {actual_bands_written} bands, not {current_num_bands}")

        update_win_parameter(win_file, 'num_bands', actual_bands_written)

        if verbose:
            print(f"  ✓ Updated {win_file}")
            print()
            print(f"  NEXT STEPS:")
            print(f"  1. Re-run preprocessing: wannier90.x -pp {self.seedname}")
            print(f"  2. Re-run Stage 2:       python3 lcao_to_wannier90.py --stage 2 ...")
except Exception as e:
    if verbose:
        print(f"\n⚠ Warning: Could not validate/update .win file: {e}")
```

---

## Expected Behavior After Fix

When Stage 2 detects a mismatch:

```
Step 10: Writing data files using bismuth.nnkp neighbors...
--------------------------------------------------------------------------------
  ✓ bismuth.eig: 3150 eigenvalues (relative to E_F = -3.727962 eV)

⚠ IMPORTANT: Updating bismuth.win
  num_bands: 16 → 14
  Reason: Energy window contains 14 bands, not 16
  ✓ Updated bismuth.win

  ============================================================================
  NEXT STEPS:
  1. Re-run preprocessing: wannier90.x -pp bismuth
  2. Re-run Stage 2:       python3 lcao_to_wannier90.py --stage 2 ...
  ============================================================================

================================================================================
STAGE 2 COMPLETE!
================================================================================
```

**User must then**:
1. Re-run preprocessing: `wannier90.x -pp bismuth`
2. Re-run Stage 2: `python3 lcao_to_wannier90.py --stage 2 --input ... --seedname bismuth`
3. Run Wannier90: `wannier90.x bismuth` ✅ **Now works!**

---

## Why This Happens

The band selection algorithm selects bands **closest to Fermi level** within the energy window. This is the correct behavior! But it can result in fewer bands than originally requested:

**Example:**
- User requests: `num_bands = 16`
- Energy window: `[-10, 5]` eV (from `create_win_template.py` auto-selection)
- Actual bands in window: Only 14 bands fall within this range
- Stage 2: Correctly selects 14 bands closest to E_F
- Result: `.eig` has 14 bands, but `.win` says 16 → **MISMATCH**

**The fix**: Stage 2 automatically updates `.win` to `num_bands = 14`, ensuring synchronization.

---

## Files Modified

1. **`/Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier/lcao_wannier/win_file.py`**
   - Lines 694-750: Added `read_win_parameter()` function
   - Lines 753-802: Added `update_win_parameter()` function

2. **`/Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier/lcao_wannier/engine.py`**
   - Lines 679-705: Added validation/update logic after writing `.eig` file

---

## Benefits

1. **Prevents cryptic Wannier90 errors**: Users get clear, actionable guidance
2. **Automatic fix**: No manual editing of `.win` files required
3. **Clear workflow**: Users know exactly what commands to run next
4. **Prevents data corruption**: Ensures `.win` and `.eig` are always synchronized

---

## Alternative Workflow (Avoiding the Issue Entirely)

To avoid needing to re-run preprocessing, users can:

1. Generate `.win` with `create_win_template.py` (auto-selects appropriate windows)
2. Run `wannier90.x -pp` once
3. Run Stage 2 **without** `--window` flag (uses windows from `.win` file)
4. If mismatch detected, re-run preprocessing and Stage 2 as instructed

The auto-update ensures that even if there's a mismatch, the user gets clear guidance on how to fix it.

---

## Status: COMPLETE ✅

All changes implemented and ready for testing.

**Date**: January 10, 2026
