# Correct Two-Stage Workflow for LCAO-to-Wannier90

## Overview

The proper workflow has **3 steps** with Wannier90 preprocessing in between:

1. **Stage 1**: Generate `.win` file from CRYSTAL output
2. **Preprocessing**: Run `wannier90.x -pp` to generate `.nnkp` (neighbor list)
3. **Stage 2**: Generate `.eig`, `.amn`, `.mmn` using the `.nnkp` neighbor information

This ensures the MMN file uses exactly the neighbors Wannier90 expects.

---

## Complete Command Sequence

### Step 1: Stage 1 - Generate .win File

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier
source venv/bin/activate

python3 lcao_to_wannier90.py \
    --stage 1 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_improved \
    --window -8.5 0.5 \
    --projections "Bi:p"
```

**Output:**
- `bismuth_improved.win` - Wannier90 parameter file

**What this does:**
- Parses CRYSTAL output
- Creates SOC spin-block matrices
- Analyzes bands in the energy window
- Generates `.win` file with proper settings

---

### Step 2: Edit .win for Better Localization

Edit `bismuth_improved.win` to add disentanglement:

```fortran
! Change these lines:
num_wann = 12        ! Target 12 WFs (6 p-orbitals × 2 spinor)
num_bands = 16       ! Include more bands for disentanglement

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

! Projections should already be:
begin projections
Bi:p
end projections
```

**Important:**
- The `Bi:p` projection creates **6 orbital projections** (px, py, pz on 2 atoms)
- With `spinors = .true.`: 6 × 2 = **12 WFs**
- Disentanglement extracts optimal 12-dimensional subspace from 16 bands

---

### Step 3: Run Wannier90 Preprocessor (on cluster)

Transfer `.win` to cluster and run preprocessor:

```bash
# Transfer to cluster
scp bismuth_improved.win cluster:~/wannier_test/

# On cluster
ssh cluster
cd wannier_test
wannier90.x -pp bismuth_improved
```

**Output:**
- `bismuth_improved.nnkp` - Neighbor k-point file

**What this does:**
- Reads your `.win` file
- Calculates reciprocal lattice
- Determines optimal b-vector shells
- Generates neighbor list for each k-point
- **This is essential for correct MMN generation!**

---

### Step 4: Transfer .nnkp Back to Local Machine

```bash
# On local machine
scp cluster:~/wannier_test/bismuth_improved.nnkp ./
```

---

### Step 5: Stage 2 - Generate .eig, .amn, .mmn

Now generate the data files using the `.nnkp`:

```bash
python3 lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_improved \
    --window -8.5 0.5
```

**Output:**
- `bismuth_improved.eig` - Eigenvalues
- `bismuth_improved.amn` - Projection matrices (overlap-corrected: A = S@C)
- `bismuth_improved.mmn` - Overlap matrices (phase-corrected with degenerate k-point handling)

**What this does:**
- Reads `.nnkp` neighbor list
- Parses atomic basis info for phase correction
- Doubles basis_atom_map for SOC
- Solves eigenvalue problems
- Writes `.eig`, `.amn` with overlap correction
- Writes `.mmn` with atomic center phase correction

**Critical:** The MMN writer will:
- Use neighbors from `.nnkp` (not internally generated)
- Apply phase correction: `exp(-i*b·τ)` for each neighbor
- Handle degenerate k-points (when k+b = k in 2D)

---

### Step 6: Transfer All Files to Cluster

```bash
scp bismuth_improved.* cluster:~/wannier_test/
```

---

### Step 7: Run Wannier90 on Cluster

```bash
# On cluster
ssh cluster
cd wannier_test
wannier90.x bismuth_improved
```

---

### Step 8: Check Results

```bash
# Check final spreads
grep "Final Spread" bismuth_improved.wout

# Check convergence
grep "CONV" bismuth_improved.wout | tail -20

# Check Omega breakdown
grep "Omega" bismuth_improved.wout | grep "SPRD" | tail -10
```

**Expected results:**
```
Omega I  =    3.57 Ang²     (gauge-invariant, won't change much)
Omega D  =   30-80 Ang²     (should decrease from 155)
Omega OD =  <100 Ang²       (should decrease from 559!)
Omega Total = <200 Ang²     (should decrease from 718)
```

---

## Troubleshooting

### Issue: "Could not read .nnkp file"
**Solution:** Make sure you ran `wannier90.x -pp` and the `.nnkp` exists in the current directory.

### Issue: Still getting large Omega_OD > 300 Ang²
**Solutions:**
1. Try different projections: `Bi:sp`, `Bi:s`, `Bi:sp3`
2. Adjust disentanglement window
3. Increase num_bands to 18-20

### Issue: "num_wann mismatch"
**Solution:** Remember with spinors:
- `Bi:p` → 6 projections × 2 = 12 WFs
- If you want 10 WFs, use manual projections (see below)

### Manual Projections for 10 WFs (no disentanglement)

If you want exactly 10 WFs without disentanglement:

```fortran
num_wann = 10
num_bands = 10

begin projections
f=0.333298,0.166649,0.001797:px
f=0.333298,0.166649,0.001797:py
f=0.333298,0.166649,0.001797:pz
f=-0.333298,-0.166649,-0.001797:px
f=-0.333298,-0.166649,-0.001797:py
end projections
```
(5 projections × 2 spinor = 10 WFs)

---

## Summary: Why This Workflow?

1. **Stage 1** creates `.win` so you can configure projections/settings
2. **Preprocessing** determines optimal b-vectors for your system
3. **Stage 2** generates data files using those exact b-vectors
4. **Result:** MMN overlaps calculated for the right neighbors!

**Don't skip the preprocessing step!** Without the `.nnkp`, the code generates neighbors internally, which may not match what Wannier90 expects.
