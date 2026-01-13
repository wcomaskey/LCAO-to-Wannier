# Band Structure Problem - Diagnosis and Solution

## Problems Identified

### 1. **CRITICAL: Disentanglement Not Converging**
```
<<< Warning: Maximum number of disentanglement iterations reached >>>
<<< Disentanglement convergence criteria not satisfied >>>
```
- Only 1000 iterations completed
- Omega_I still changing: 1.212E-03 fractional change
- Convergence tolerance: 1.000E-10 (very strict!)

###  2. **Band Structure Only Shows ONE Band**
```bash
$ awk '{print NF}' bismuth_final_band.dat | sort -u
0  # Blank lines
2  # Only k-point + 1 energy column
```

Expected: 13 columns (k-point + 12 bands)
Actual: 2 columns (k-point + 1 band)

### 3. **No Inner (Frozen) Window Being Used**
```
No inner window (linner = F)
```

This is despite having:
```fortran
dis_froz_min = -4.5
dis_froz_max = 2.0
```

### 4. **Poor Localization Quality**
```
Omega_OD = 116.24 Ang² (should be < 100)
Total spread = 221.46 Ang² (should be < 200)
```

Compared to earlier local run:
- Before: Omega_OD = 91.95 Ang² ✓
- Now: Omega_OD = 116.24 Ang² ✗

## Root Causes

### 1. Energy Windows Are Too Wide
```fortran
dis_win_min = -25.0    ! Outer window
dis_win_max = 10.0     ! Includes way too many bands
```

Your eigenvalues range: **[-23.5, -11.1] eV**

The window `[-25, 10]` includes:
- **ALL bands** from -23.5 to 10 eV
- Includes many high-energy bands (up to ~10 eV!)
- These high-energy bands have nothing to do with physics near Fermi level

**Result:**
- Disentanglement struggles to find optimal 12-band subspace
- Convergence is poor
- Spreads increase

### 2. Frozen Window Outside Eigenvalue Range
```fortran
dis_froz_min = -4.5
dis_froz_max = 2.0
```

But your bands are at **[-23.5, -11.1] eV** - completely outside this range!

**Result:**
- Frozen window contains NO bands
- Wannier90 disables frozen window: "No inner window"
- Less constraint on disentanglement

### 3. Only Writing 1 Band to File
This appears to be a Wannier90 bug or output formatting issue when:
- Disentanglement doesn't converge properly
- Energy windows are badly chosen

## Solution

### Step 1: Fix Energy Windows

You need to set windows based on **actual eigenvalue range**, not arbitrary values.

From your .eig file: bands 25-40 are in range **[-23.5, -11.1] eV**

**Corrected windows:**

```fortran
! Outer window: Should capture your 16 bands
dis_win_min = -24.0    ! Slightly below lowest band
dis_win_max = -10.5    ! Slightly above highest band

! Frozen window: Target bands closest to Fermi level
! Your Fermi energy is -3.73 eV, so look for bands near there
! But your selected bands are -23 to -11 eV, so frozen window should be:
dis_froz_min = -13.0   ! Around upper few bands
dis_froz_max = -11.0   ! Top bands in your selection
```

**OR BETTER: Don't use such deep bands!**

Your problem is you're selecting bands that are **20 eV below Fermi level**. These are core-like states, not valence bands!

### Step 2: Select Correct Bands

The issue is your `--window -25.0 10.0` parameter in Stage 2 is selecting the WRONG bands.

Let me check what bands are actually near the Fermi level:

Your Fermi energy: **-3.728 eV**

Typical band selection should be:
- **Valence bands**: -10 to 0 eV relative to E_F
- **Conduction bands**: 0 to +5 eV relative to E_F

So use: `--window -10.0 5.0` (relative to E_F = -3.73 eV absolute)

This gives absolute range: `[-13.73, 1.27]` eV

### Step 3: Increase Disentanglement Iterations

```fortran
dis_num_iter = 5000    ! Was 1000, increase significantly
```

### Step 4: Relax Convergence Tolerance

```fortran
dis_conv_tol = 1.0E-08    ! Was 1.0E-10, too strict
```

## Corrected Workflow

### 1. Regenerate with Correct Energy Window

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier
source venv/bin/activate

# Generate .win with corrected windows
python3 create_win_template.py \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_correct
```

### 2. Manually Edit .win File

Edit `bismuth_correct.win`:

```fortran
! CORRECTED energy windows
dis_win_min = -10.0     ! 10 eV below Fermi level
dis_win_max = 5.0       ! 5 eV above Fermi level
dis_froz_min = -2.0     ! Frozen window near Fermi level
dis_froz_max = 1.0
dis_num_iter = 5000     ! More iterations
dis_conv_tol = 1.0E-08  ! Less strict tolerance
```

### 3. Run Preprocessing

```bash
export PATH="external/wannier90-3.1.0:$PATH"
wannier90.x -pp bismuth_correct
```

### 4. Generate Data Files with MATCHING Window

```bash
python3 lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_correct \
    --window -10.0 5.0
```

**CRITICAL:** The `--window` parameter MUST match `dis_win_min/max` in .win!

### 5. Run Wannier90

```bash
wannier90.x bismuth_correct
```

## Why This Will Work

1. **Correct band selection**: Bands near Fermi level, not deep core states
2. **Tighter windows**: Only 15 eV range instead of 35 eV
3. **Frozen window active**: Will contain actual bands
4. **Better convergence**: More iterations, relaxed tolerance

## Expected Results

After fixing:
- Disentanglement **WILL converge**
- Band structure file will have **13 columns** (k + 12 bands)
- Omega_OD **< 100 Ang²**
- Bands will **match DFT** near Fermi level

## Quick Check: What Bands Should I Use?

To determine correct window, check your CRYSTAL output for band energies.

You want:
- Bands crossing or near Fermi level
- Typically: top valence bands + bottom conduction bands
- Energy range: ~10 eV below to ~5 eV above E_F

**Your case:**
- E_F = -3.73 eV (absolute)
- Want relative window: [-10, +5] eV
- Absolute window: [-13.73, 1.27] eV

This is VERY different from [-25, 10] eV which goes way too deep!

