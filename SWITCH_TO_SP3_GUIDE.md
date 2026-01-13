# Switching to sp³ Projections

## Changes Required

### 1. Update .win File

Change these parameters in `bismuth_final.win`:

```fortran
num_wann = 16        ! Was 12
num_bands = 20       ! Was 16 (or use 22 for more flexibility)

! Widen energy window to capture 20 bands
dis_win_min = -26.0  ! Was -25.0
dis_win_max = -8.0   ! Was 10.0

! Adjust frozen window for bands near Fermi level
dis_froz_min = -15.0
dis_froz_max = -10.0

! Change projections
begin projections
Bi:sp3               ! Was Bi:p
end projections
```

### 2. Rerun Preprocessing (REQUIRED!)

```bash
export PATH="external/wannier90-3.1.0:$PATH"
wannier90.x -pp bismuth_final
```

This regenerates `bismuth_final.nnkp` with:
- 16 Wannier functions (not 12)
- 20 bands (not 16)
- sp³ projection definitions

### 3. Regenerate Data Files with New Window

```bash
source venv/bin/activate

python3 lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_final \
    --window -26.0 -8.0
```

**IMPORTANT:** Match the `--window` parameter to the `dis_win_min/max` in .win file!

### 4. Run Wannier90

```bash
wannier90.x bismuth_final
```

## Expected Results

With sp³ projections:
- **16 Wannier functions** (8 sp³ projections × 2 spinor)
- **20 bands** for disentanglement
- May get **better localization** than Bi:p if s-orbitals contribute significantly

## Why sp³ Might Be Better

For Bismuth:
- **p-orbitals only (Bi:p)**: Captures p-band character
- **sp³ hybrid (Bi:sp³)**: Captures both s and p character
  
If your bands near the Fermi level have **s-p hybridization**, sp³ projections will give:
- Better initial guess
- Faster convergence
- Possibly lower spreads

## How to Decide: Bi:p vs Bi:sp³

**Use Bi:p if:**
- Bands are primarily p-character
- You want to isolate p-bands only
- Current results are already good (Omega_OD < 100)

**Use Bi:sp³ if:**
- Bands have s-p mixing
- Current localization is poor
- You want to capture both s and p bands

## Current Status: Bi:p Results

Your current Bi:p results are **already excellent**:
- Omega_OD = 91.95 Ang² (< 100 target ✓)
- Total spread = 174.18 Ang² (< 200 target ✓)
- Most WFs < 20 Ang² (well-localized ✓)

**Recommendation:** Keep Bi:p unless you have a specific reason to include s-orbitals!

## Alternative: Try Both and Compare

You can keep both versions:

```bash
# Current Bi:p version (12 WFs)
bismuth_final.*

# New Bi:sp3 version (16 WFs)
bismuth_sp3.*
```

Then compare:
1. Convergence speed
2. Final spreads
3. Band structure accuracy vs DFT

