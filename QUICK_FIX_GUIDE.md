# Quick Fix Guide - Get Working Files Now

## Problem
The automatic test has a bug with neighbor list conversion. Here's the quickest path to working files.

## Solution: Two-Step Process

### Step 1: Generate Initial Files (will be incomplete)

```bash
cd /path/to/LCAO-to-Wannier
source venv/bin/activate

python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -5.0 3.0 \
    --seedname bismuth_improved \
    --output-dir test_output
```

This will create `.amn`, `.eig`, and `.win` files (the `.mmn` may be incomplete, but that's okay).

### Step 2: Edit the .win File

Edit `test_output/bismuth_improved.win`:

```fortran
! Change these parameters:
num_wann = 12
num_bands = 16

! Add after num_bands:
dis_win_min = -8.5
dis_win_max = 0.5
dis_froz_min = -8.0
dis_froz_max = -1.1
dis_num_iter = 1000
dis_mix_ratio = 0.5

! Increase iterations:
num_iter = 5000
conv_window = 5

! Change projections:
begin projections
Bi:p
end projections
```

### Step 3: Run Wannier90 Preprocessor (on cluster)

Transfer the .win file to your cluster:

```bash
scp test_output/bismuth_improved.win cluster:~/test/
```

On cluster:

```bash
ssh cluster
cd test
wannier90.x -pp bismuth_improved
```

This creates `bismuth_improved.nnkp` with proper neighbors for your system.

### Step 4: Copy .nnkp Back and Regenerate

```bash
# On local machine:
scp cluster:~/test/bismuth_improved.nnkp test_output/

# Now regenerate - this time it will use the .nnkp file:
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -5.0 3.0 \
    --seedname bismuth_improved \
    --output-dir test_output
```

Now the `.mmn` file will be generated correctly using the neighbors from `.nnkp`.

### Step 5: Transfer All Files and Run

```bash
scp test_output/bismuth_improved.* cluster:~/test/
ssh cluster
cd test
wannier90.x bismuth_improved
```

### Step 6: Check Results

```bash
grep "Final Spread" bismuth_improved.wout
grep "Omega" bismuth_improved.wout | grep "SPRD" | tail -10
```

## Expected Results

With `Bi:p` projections and disentanglement:
- **Omega_OD:** Should drop from 559 → **50-150 Ang²**
- **Individual spreads:** Should drop from 40-90 → **10-30 Ang²**
- **Total spread:** Should drop from 718 → **100-250 Ang²**

## If Results Are Still Poor

1. **Check what bands were actually selected:**
   ```bash
   grep "Wannierising" bismuth_improved.wout
   ```

2. **Try different projection types:**
   - `Bi:s` (s-orbitals only)
   - `Bi:sp` (s + p mixed)
   - `Bi:sp3` (sp3 hybrids)

3. **Adjust disentanglement window:**
   ```fortran
   dis_win_min = -9.0  ! Expand window
   dis_win_max = 1.0
   ```

## Alternative: Use Existing Files

If you already have working `.amn`, `.eig`, `.mmn` files from the original test (test_output/bismuth_test.*), you can:

1. Copy them:
   ```bash
   cp test_output/bismuth_test.{amn,eig,mmn,nnkp} test_output/bismuth_improved.
   ```

2. Just edit the .win file with the settings above

3. Run wannier90.x directly on cluster

The `.amn` and `.mmn` already have the phase corrections - you just need better projections in the .win file!
