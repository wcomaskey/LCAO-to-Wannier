# Diagnosis: Negative Omega_I and Poor Localization

## The Problem

After running Wannier90 with the validated configuration (`num_wann=12`, `num_bands=12`, window `[-5.7, 5.7]` eV):

```
Initial State:
  Omega_I      =    -73.134503 Ang²  ← NEGATIVE! (Should be positive)
  Omega_D      =     70.948675 Ang²
  Omega_OD     =    526.413011 Ang²
  Omega_Total  =    524.227182 Ang²
```

**Critical Issue**: **Negative Omega_I** indicates a fundamental problem with the Wannier function construction.

## What Is Omega_I?

- **Omega_I** (gauge-invariant spread): Spread that cannot be reduced by unitary transformations
- **Must ALWAYS be positive** for physical Wannier functions
- Negative value → mathematical inconsistency in the input data

## Possible Causes

### 1. **Incorrect Band Selection** (Most Likely)
**Status**: Bands 40-51 selected from 112 total bands

**Issue**: For Bismuth with SOC (112 bands total), selecting bands 40-51 might be:
- Too deep in valence band
- Missing relevant conduction bands
- Not capturing the p-orbital character we need

**Evidence**: The energy window `[-5.7, 5.7]` eV only contains 12 bands total, and we're using all of them. This might not be the right 12 bands!

### 2. **Wrong Projections**
**Current**: Using `Bi:p` automatic projections (6 orbitals × 2 spinors = 12 WFs)

**Issue**: The automatic projection might not match the actual character of bands 40-51.

### 3. **Non-orthogonal Basis Issue**
**Status**: LCAO uses non-orthogonal basis functions

**Issue**: The transformation from non-orthogonal LCAO to orthogonal Wannier90 input might have errors in:
- Overlap matrix handling
- Hamiltonian transformation
- Band eigenstate construction

### 4. **`.amn` File Corruption**
The projection matrix (`.amn`) might have incorrect values, leading to poor initial guess.

## Diagnostic Steps

### Step 1: Check Which Bands We're Actually Using

```bash
# Check the Stage 2 log
grep -A 20 "Selected band indices" <stage2_log>
```

**Current**: Bands 40-51 selected

**Question**: Are these the p-orbital derived surface states, or something else?

### Step 2: Try Different Energy Window

The key insight: **We need to find which 12 bands correspond to Bi p-orbitals**

Try a **wider window** that captures more bands, then let disentanglement select the right 12:

```
num_wann = 12
num_bands = 18  # More than 12 for flexibility
dis_win_min = -10.0
dis_win_max = 6.0
dis_froz_min = -4.0
dis_froz_max = 3.0
```

This gives Wannier90 flexibility to **disentangle** the p-character from mixed states.

### Step 3: Manual Band Selection

Instead of auto-selection, manually specify which bands to use based on:
1. Orbital character analysis
2. Band structure around Fermi level
3. Expected physics (Bi p-orbitals forming surface states)

## Recommended Fix

### Option A: Use Wider Window with Disentanglement

```bash
# 1. Create new .win with wider window
python3 validate_and_create_win.py \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --num-wann 12

# 2. Manually edit bismuth.win:
#    dis_win_min = -10.0
#    dis_win_max = 6.0
#    num_bands = 18  # or however many bands fall in this window

# 3. Re-run workflow
./external/wannier90-3.1.0/wannier90.x -pp bismuth
python3 lcao_to_wannier90.py --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -10.0 6.0
```

### Option B: Check Band Character First

Before running Wannier90, we need to understand **which bands have p-orbital character**.

This requires:
1. Plotting band structure with orbital projections
2. Identifying which band indices correspond to Bi-p states
3. Using those specific bands in `.win` file

### Option C: Try sp³ Projections Instead

If Bi surface reconstruction involves sp³ hybridization rather than pure p:

```
# In bismuth.win, replace:
begin projections
Bi:p
end projections

# With:
begin projections
Bi:sp3
end projections
```

This gives 4 orbitals × 2 spinors × 2 atoms = 16 WFs, so set `num_wann = 16`.

## What To Do Next

**Immediate Action**: The negative Omega_I means the current configuration is fundamentally broken. You need to:

1. **Stop using the current files** - they won't converge to meaningful Wannier functions

2. **Determine correct band indices** - Need to know which of the 112 bands are actually Bi p-derived surface states

3. **Re-run with correct configuration**

## Questions to Answer

1. **What is the expected band structure?**
   - How many Bi p-derived bands should cross the Fermi level?
   - Are we looking for surface states or bulk states?

2. **What window captures the right bands?**
   - Need actual band structure analysis
   - Can't just guess energy windows

3. **Are the projections correct?**
   - Is `Bi:p` the right choice?
   - Should we use `Bi:sp3` or manual orbital specifications?

## Files to Check

1. **`validated_run_final.log`**: See exact band indices selected
2. **`bismuth.wout`**: Check for other warnings/errors
3. **Stage 2 output**: Verify band selection logic

---

**Bottom Line**: The negative Omega_I is a red flag that we're using the wrong bands or wrong projections. We need to go back and properly identify which bands correspond to the Bi p-orbitals you want to Wannierize.

**Date**: January 10, 2026
