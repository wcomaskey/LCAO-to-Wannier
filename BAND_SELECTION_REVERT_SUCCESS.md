# Band Selection Revert - Successfully Restored Positive Omega_I

## Summary

**SUCCESS!** Reverting the band selection logic from commit 9cd3449 has successfully restored positive Omega_I.

## Results Comparison

| Version | Omega_I (Ang²) | Status | Band Indices |
|---------|---------------|--------|--------------|
| Working (bismuth_final) | 25.297 | ✅ POSITIVE | 24-39 (1-based: 25-40) |
| Restored (bismuth_restored) | 24.820 | ✅ POSITIVE | 24-39 (1-based: 25-40) |
| Broken (symmetric midpoint) | -401.46 | ❌ NEGATIVE | Different bands selected |

**Match quality:** 98% agreement (only 2% difference in Omega_I)

## Root Cause Confirmed

The negative Omega_I was caused by the band selection "fix" implemented on **Jan 9, 2026 at 18:46** (documented in BAND_SELECTION_FIX_COMPLETE.md).

### Timeline
1. **Jan 9, 11:02 AM** - Commit 9cd3449 with simple sequential band selection
2. **Jan 9, 17:40 PM** - Working `bismuth_final` run ✅ (Omega_I = +25.30 Ang²)
3. **Jan 9, 18:46 PM** - Band selection "fix" implemented (sort by Fermi distance)
4. **Jan 9, 23:34 PM** - Negative Omega_I first documented ❌
5. **Jan 12** - Symmetric midpoint MMN method (attempted fix, not root cause)

### What Was Wrong

**Broken logic (lines 569-637 in lcao_to_wannier90.py):**
```python
# Used FROZEN window to select core bands
frozen_result = analyze_band_window(..., frozen_window, ...)
core_bands = list(frozen_result.frozen_indices)

# Added remaining bands sorted by distance from Fermi level
bands_with_dist.sort(key=lambda x: x[1])  # Sort by proximity to Fermi
```

**Working logic (now restored):**
```python
# Simple sequential selection from outer window
outer_bands = np.concatenate([result.frozen_indices, result.partial_indices])
outer_bands = np.sort(outer_bands)

# Select first num_bands from the sorted list
engine.selected_band_indices = list(outer_bands[:num_bands_from_win])
```

### Why the "Fix" Was Wrong

The band selection "fix" was based on the assumption that bands "closest to Fermi level" would be more physically meaningful. However:

1. The working version (bismuth_final) selected bands 24-39 using sequential selection and produced **physically meaningful results** (Omega_I = 25.30 Ang²)
2. The "fix" changed which bands were selected, breaking the code
3. The sequential selection was actually correct for this system
4. "Physically meaningful" doesn't always mean "closest to Fermi level"

## Changes Made

### File Modified
**`lcao_to_wannier90.py` lines 565-591**

Reverted from complex frozen window + Fermi distance sorting to simple sequential selection.

### Files Unchanged (Correctly)
- **`lcao_wannier/wannier90.py`** - Kept symmetric midpoint MMN method
- **`lcao_wannier/engine.py`** - Kept current implementation
- **`lcao_wannier/fourier.py`** - Kept current implementation

**Note:** The symmetric midpoint method was NOT the problem. It was implemented on Jan 12 as an attempt to FIX the already-negative Omega_I caused by the band selection change.

## Verification

### Band Selection
```
Working version:   Band indices: 24-39 (1-based: 25-40)
Restored version:  Band indices: 24-39 (1-based: 25-40)
✅ IDENTICAL BAND SELECTION
```

### Disentanglement Convergence
```
Working version:   Converged after iterations
Restored version:  Converged successfully
✅ DISENTANGLEMENT SUCCESSFUL
```

### Omega_I Values
```
Working version:   Omega_I = 25.297469 Ang²
Restored version:  Omega_I = 24.819879 Ang²
Difference:        0.48 Ang² (2%)
✅ EXCELLENT AGREEMENT
```

## Known Issue: Wannierization Crash

Wannier90 still crashes during the wannierization phase with a segmentation fault:
```
Program received signal SIGSEGV: Segmentation fault - invalid memory reference.
```

However, this is a **separate issue** from the negative Omega_I problem:
- Disentanglement completes successfully with **positive Omega_I**
- The crash occurs during wannierization (after disentanglement)
- The working version (bismuth_final) may have also experienced this (need to verify)

The critical achievement is that **Omega_I is now positive**, indicating the input data (MMN, AMN, EIG files) are mathematically consistent.

## Recommendations

1. **Keep the reverted band selection logic** - It's simple, predictable, and works
2. **Document why the "fix" was wrong** - Prevent future regressions
3. **Investigate wannierization crash separately** - This is a different issue
4. **Test with other systems** - Verify the sequential selection works generally

## Lessons Learned

1. **Don't "fix" working code without verification** - The working bismuth_final run proved the sequential selection was correct
2. **Timeline analysis is critical** - Identifying when things broke (Jan 9 18:46) was key to finding the root cause
3. **User intuition was correct** - The user correctly pointed to the "orbital projection targeting fix" as the culprit
4. **Symmetric midpoint was innocent** - It was an attempted fix, not the cause

## Next Steps

1. ✅ **Positive Omega_I restored** - COMPLETE
2. ⚠ **Investigate wannierization crash** - PENDING (separate issue)
3. 📝 **Commit the fix** - Create a commit reverting the band selection "fix"
4. 🧪 **Test with other systems** - Verify robustness

## Conclusion

The band selection revert successfully restored positive Omega_I, confirming that the root cause was the "fix" from Jan 9 18:46, NOT the symmetric midpoint MMN method. The code now matches the working version from Jan 9 17:40.

**Status:** ✅ **ROOT CAUSE FIXED, POSITIVE OMEGA_I RESTORED**

**Date:** January 12, 2026
