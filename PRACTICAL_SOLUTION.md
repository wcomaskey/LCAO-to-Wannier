# Practical Solution - Work with Available Bands

## The Core Problem

Your CRYSTAL output only contains 16 bands, and they are:
- **Bands 1-12**: ~23 eV below Fermi level (deep core states)
- **Bands 13-14**: ~14 eV below Fermi level  
- **Bands 15-16**: ~10 eV below Fermi level

**None of these are valence or conduction bands near the Fermi level!**

This is why your band structure looks terrible - you're plotting core states, not the electronic states relevant for material properties.

## Why This Happened

When you set `--window -25.0 10.0` in Stage 2, the code selected the first 16 bands it could find in that window, which turned out to be these deep core states.

## Solutions

### Option 1: Rerun CRYSTAL with Correct Bands (RECOMMENDED)

You need to go back to your CRYSTAL calculation and:

1. **Increase number of bands** calculated (e.g., ask for 40-50 bands)
2. **Make sure bands cross the Fermi level** (check CRYSTAL output)
3. Then regenerate Wannier90 files

This is the **correct** solution for getting meaningful band structure.

### Option 2: Work with What You Have (Temporary)

If you can't rerun CRYSTAL, you can still make Wannier90 work with these 16 bands, but understand:
- The band structure will show **core states**, not valence/conduction bands
- This is NOT useful for materials science
- But it can verify the workflow is working

**Corrected settings for your current 16 bands:**

```fortran
! Energy windows matching ACTUAL band range
dis_win_min = -24.0      ! Captures all 16 bands
dis_win_max = -9.5       ! Upper edge of band 16

! Frozen window around highest bands
dis_froz_min = -11.0     ! Around bands 15-16
dis_froz_max = -9.5      ! Top of band range

! Relaxed convergence
dis_num_iter = 5000
dis_conv_tol = 1.0E-08   ! Less strict
```

## Workflow to Fix Current Setup

### Step 1: Edit bismuth_final.win

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Edit the .win file with correct windows
```

Edit these lines in `bismuth_final.win`:

```fortran
dis_win_min = -24.0
dis_win_max = -9.5
dis_froz_min = -11.0  
dis_froz_max = -9.5
dis_num_iter = 5000
dis_conv_tol = 1.0E-08
```

### Step 2: Rerun Preprocessing

```bash
export PATH="external/wannier90-3.1.0:$PATH"
wannier90.x -pp bismuth_final
```

### Step 3: Regenerate Data Files

```bash
source venv/bin/activate

python3 lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_final \
    --window -24.0 -9.5
```

**IMPORTANT:** The window must match the .win file!

### Step 4: Run Wannier90

```bash
wannier90.x bismuth_final
```

This should:
- ✅ Disentangle properly
- ✅ Output all 12 bands
- ✅ Give positive spreads

But remember: **the bands are core states**, not useful physics!

## What You Should Really Do

### Check Your CRYSTAL Input

Look at your CRYSTAL input file. You likely have a line like:

```
SHRINK
15 15 1
END
```

And somewhere a band calculation section. You need to add more bands.

### Typical CRYSTAL Settings for Band Structure

```
BAND
TITLE
Bismuth band structure
```SHRINK
15 15
EIGENVECTORS
40    ! Number of eigenvectors (bands) to calculate
FERMI
END
```

Ask for at least 40 bands to ensure you get valence + conduction bands near E_F.

## Summary

**Root cause:** Your CRYSTAL calculation only has 16 bands, all of which are 10-23 eV below Fermi level (core states).

**Quick fix for workflow testing:** Use windows `[-24.0, -9.5]` to match your actual bands.

**Proper solution:** Rerun CRYSTAL with 40+ bands to get valence/conduction bands near Fermi level, then use windows like `[-10, +5]` eV relative to E_F.

**The band structure will only be meaningful once you have bands crossing the Fermi level!**

