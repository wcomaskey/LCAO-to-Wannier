# Instructions for Running Localization Tests

## Overview

We have two test configurations ready to improve Wannier localization:
- **Path 1**: 12 WFs with automatic `Bi:p` projections and disentanglement (RECOMMENDED)
- **Path 2**: 10 WFs with manual projections, no disentanglement

Both use the corrected phase-MMN and overlap-AMN from the bug fix.

## Current Status

✅ **Bug Fixed**: Negative spreads eliminated (now positive!)
❌ **Localization Poor**: Omega_OD = 559 Ang² (too large)
🎯 **Goal**: Omega_OD < 100 Ang², individual spreads < 20 Ang²

## Quick Setup (Manual - Recommended)

Since the automatic script has some dependency issues, here's the manual approach:

### Step 1: Generate Base Files

Run the working test to generate `.amn`, `.mmn`, `.eig`:

```bash
cd /path/to/LCAO-to-Wannier
source venv/bin/activate

# Generate for 12 WFs (wider window to include more bands)
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -8.5 0.5 \
    --seedname bismuth_path1_auto \
    --output-dir test_output
```

This creates:
- `bismuth_path1_auto.amn` (projection matrices)
- `bismuth_path1_auto.mmn` (overlap matrices - phase corrected!)
- `bismuth_path1_auto.eig` (eigenvalues)
- `bismuth_path1_auto.win` (Wannier90 input - needs editing)

### Step 2: Edit .win File for Path 1 (12 WFs + Disentanglement)

Edit `test_output/bismuth_path1_auto.win`:

```fortran
! Change these lines:
num_wann = 12        ! Target 12 WFs (6 p-orbitals × 2 spinor)
num_bands = 16       ! Include bands 40-55 (16 bands)

! Add disentanglement (after num_bands):
dis_win_min = -8.5   ! Outer window min
dis_win_max = 0.5    ! Outer window max
dis_froz_min = -8.0  ! Frozen window min
dis_froz_max = -1.1  ! Frozen window max
dis_num_iter = 1000
dis_mix_ratio = 0.5

! Change iterations:
num_iter = 5000

! Change projections from 'random' to:
begin projections
Bi:p
end projections
```

### Step 3: Run wannier90 Preprocessor

```bash
cd test_output
wannier90.x -pp bismuth_path1_auto
```

This generates `bismuth_path1_auto.nnkp` with proper neighbor list.

### Step 4: Regenerate .mmn with Correct Neighbors

```bash
cd ..
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -8.5 0.5 \
    --seedname bismuth_path1_auto \
    --output-dir test_output
```

Now the `.mmn` file will use neighbors from the `.nnkp` file.

### Step 5: Transfer to Cluster and Run

```bash
# Copy all files
scp test_output/bismuth_path1_auto.* cluster:~/localization_test/

# On cluster:
ssh cluster
cd localization_test
wannier90.x bismuth_path1_auto
```

### Step 6: Check Results

```bash
# On cluster, check convergence:
grep "CONV" bismuth_path1_auto.wout | tail -20

# Check final spreads:
grep "Final Spread" bismuth_path1_auto.wout

# Check Omega_OD:
grep "Omega OD" bismuth_path1_auto.wout | tail -5
```

**Expected improvements:**
- Omega_I: ~3.6 Ang² (gauge-invariant, won't change much)
- Omega_D: 155 → **30-80 Ang²** (should decrease)
- Omega_OD: 559 → **<100 Ang²** (major decrease expected)
- Total: 718 → **<150 Ang²**

## Path 2: Manual Projections (10 WFs)

If Path 1 doesn't work well, try manual projections:

### Step 1: Generate with narrower window

```bash
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -5.0 3.0 \
    --seedname bismuth_path2_manual \
    --output-dir test_output
```

### Step 2: Edit .win for Manual Projections

Edit `test_output/bismuth_path2_manual.win`:

```fortran
num_wann = 10
num_bands = 10  ! No disentanglement
num_iter = 5000

! Get atomic positions from atoms_frac section in same file
! Replace random projections with:
begin projections
f=0.333298,0.166649,0.001797:px
f=0.333298,0.166649,0.001797:py
f=0.333298,0.166649,0.001797:pz
f=-0.333298,-0.166649,-0.001797:px
f=-0.333298,-0.166649,-0.001797:py
end projections
```

### Step 3-6: Same as Path 1

Run preprocessor, regenerate with `.nnkp`, transfer to cluster, run, check results.

## Troubleshooting

**Q: Still getting Omega_OD > 300 Ang²?**
- Check projection orbital types match actual band character
- Try Path 1 with disentanglement (more robust)
- Consider including more bands (16-18) in disentanglement window

**Q: Projections giving wrong number of WFs?**
- Remember: Each projection × 2 (spinor) = WFs
- 5 projections → 10 WFs
- 6 projections → 12 WFs

**Q: Segfault on local Wannier90?**
- Use cluster version (local build may have issues)
- Phase-corrected `.mmn` files are valid, just run on cluster

## Expected Timeline

- Path 1 setup: ~10 minutes
- Wannier90 run on cluster: ~5-30 minutes (depends on cluster)
- Total: Should have results within 1 hour

## Success Criteria

✅ **Good localization:**
- Omega_OD < 100 Ang²
- Individual spreads < 20 Ang²
- Converges within 1000-2000 iterations

✅ **Acceptable localization:**
- Omega_OD < 200 Ang²
- Individual spreads < 40 Ang²
- Suitable for most applications

❌ **Poor localization (current):**
- Omega_OD > 500 Ang²
- Individual spreads > 70 Ang²
- Need better projections
