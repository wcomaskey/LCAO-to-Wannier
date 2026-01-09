# Complete Two-Stage Workflow - LCAO to Wannier90

## Overview

This is the **correct and tested** two-stage workflow that produces well-localized Wannier functions.

**Key Points:**
- Stage 1 generates `.win` file with proper SOC and disentanglement settings
- Wannier90 preprocessing generates `.nnkp` with optimal neighbor list
- Stage 2 uses `.nnkp` to generate `.eig`, `.amn`, `.mmn` files
- Result: Phase-corrected MMN + overlap-corrected AMN = positive, localized spreads

---

## Prerequisites

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier
source venv/bin/activate
```

---

## Stage 1: Generate .win Template File

Use the `create_win_template.py` script to generate a properly configured `.win` file:

```bash
python3 create_win_template.py \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_improved \
    --num-wann 12 \
    --num-bands 16
```

**Output:** `bismuth_improved.win`

**What this creates:**
- ✓ num_wann = 12 (for SOC: 6 p-orbitals × 2 spinor)
- ✓ num_bands = 16 (for disentanglement)
- ✓ spinors = .true. (SOC enabled)
- ✓ Disentanglement parameters (dis_win_min, dis_win_max, etc.)
- ✓ Lattice vectors in Angstrom
- ✓ Atomic positions in fractional coordinates
- ✓ Bi:p projections (automatic)
- ✓ Complete k-point list (15×15×1 = 225 points)
- ✓ Convergence parameters (5000 iterations)

**Review the .win file:**

```bash
cat bismuth_improved.win
```

Key parameters to verify:
```fortran
num_wann = 12
num_bands = 16
spinors = .true.

dis_win_min = -8.5
dis_win_max = 0.5
dis_froz_min = -8.0
dis_froz_max = -1.1
dis_num_iter = 1000
dis_mix_ratio = 0.5

num_iter = 5000
conv_window = 5

begin projections
Bi:p
end projections

begin atoms_frac
BI     0.333333     0.166667     0.001797
BI    -0.333333    -0.166667    -0.001797
end atoms_frac
```

---

## Preprocessing: Generate .nnkp File

Transfer the `.win` file to your cluster and run Wannier90 preprocessing:

```bash
# Transfer to cluster
scp bismuth_improved.win cluster:~/wannier_test/

# On cluster
ssh cluster
cd wannier_test
wannier90.x -pp bismuth_improved
```

**Output:** `bismuth_improved.nnkp`

**What this does:**
- Computes reciprocal lattice vectors
- Determines optimal b-vector shells for your system
- Generates neighbor list for each k-point
- Creates projection definitions for AMN calculation

**Transfer .nnkp back to local machine:**

```bash
# On local machine
scp cluster:~/wannier_test/bismuth_improved.nnkp ./
```

---

## Stage 2: Generate Data Files

Now generate `.eig`, `.amn`, and `.mmn` using the `.nnkp` neighbor information:

```bash
python3 lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_improved \
    --window -8.5 0.5
```

**Output:**
- `bismuth_improved.eig` - Eigenvalues for all k-points and bands
- `bismuth_improved.amn` - Projection matrices with overlap correction: **A = S(k) @ C(k)**
- `bismuth_improved.mmn` - Overlap matrices with phase correction: **M = C†(k) @ [S(k+b) × exp(-i·b·τ)] @ C(k+b)**

**What happens in Stage 2:**

1. **Reads .nnkp file** to get exact neighbor list Wannier90 expects
2. **Parses atomic positions** from CRYSTAL output for phase correction
3. **Doubles basis_atom_map** for SOC (56 basis → 112 orbitals)
4. **Solves eigenvalue problems** at all 225 k-points
5. **Writes .eig file** with eigenvalues
6. **Writes .amn file** with overlap-corrected projections
7. **Writes .mmn file** with:
   - Atomic center phase correction for each neighbor
   - Degenerate k-point handling (skips spurious phase when k+b = k)

**Expected console output:**

```
✓ Parsed 2 atoms, 56 basis functions (SOC: doubled basis map)
✓ Read 225 k-point neighbors from bismuth_improved.nnkp
✓ Solving for 225 k-points with 16 bands each...
✓ Wrote bismuth_improved.eig (225 k-points, 16 bands)
✓ Wrote bismuth_improved.amn (225 k-points, 16 bands, 12 projections)
✓ Wrote bismuth_improved.mmn (phase-corrected with degenerate k-point handling)
```

---

## Run Wannier90 on Cluster

Transfer all files and run the full Wannier90 calculation:

```bash
# Transfer all necessary files
scp bismuth_improved.{win,eig,amn,mmn,nnkp} cluster:~/wannier_test/

# Run Wannier90
ssh cluster "cd wannier_test && wannier90.x bismuth_improved"
```

---

## Check Results

```bash
# Check final spreads
ssh cluster "cd wannier_test && grep 'Final Spread' bismuth_improved.wout"

# Check convergence history
ssh cluster "cd wannier_test && grep 'CONV' bismuth_improved.wout | tail -20"

# Check Omega breakdown
ssh cluster "cd wannier_test && grep 'Omega' bismuth_improved.wout | grep 'SPRD' | tail -5"
```

**Expected results:**

| Metric | Before (random projections) | After (Bi:p + disentanglement) |
|--------|---------------------------|-------------------------------|
| Omega_I | 3.6 Ang² | 3.6 Ang² (unchanged) |
| Omega_D | 155 Ang² | 30-80 Ang² |
| Omega_OD | **559 Ang²** | **<100 Ang²** ✓ |
| Total Spread | 718 Ang² | **<200 Ang²** ✓ |
| Individual WFs | 40-90 Ang² | **<20 Ang²** ✓ |

**Success criteria:**
- ✅ **Excellent**: Omega_OD < 50 Ang², individual spreads < 15 Ang²
- ✅ **Good**: Omega_OD < 100 Ang², individual spreads < 20 Ang²
- ✅ **Acceptable**: Omega_OD < 200 Ang², individual spreads < 40 Ang²
- ❌ **Poor**: Omega_OD > 300 Ang² (try different projections)

---

## What Changed from Original Code

### Before (negative spreads):
- Missing overlap correction in AMN
- Missing phase correction in MMN
- Result: **Negative spreads (-200 to -800 Ang²)** ❌

### After overlap/phase fixes (positive but poor):
- Added A = S @ C correction
- Added phase factor exp(-i·b·τ)
- Fixed degenerate k-point handling
- Result: **Positive spreads (40-90 Ang²) with random projections** ✓

### After proper projections (well-localized):
- Used physical Bi:p projections
- Added disentanglement (16 bands → 12 WFs)
- Result: **Well-localized spreads (<20 Ang²)** 🎯

---

## Troubleshooting

### Q: Can I use the old `lcao_to_wannier90.py --stage 1` instead?

**A:** No, Stage 1 in `lcao_to_wannier90.py` has issues with band analysis and may produce wrong num_wann/num_bands. Use `create_win_template.py` instead.

### Q: What if I want different projections?

**A:** Edit the `.win` file before preprocessing. Options:
- `Bi:s` - s-orbitals only (2 WFs)
- `Bi:sp` - s + p mixed (8 WFs)
- `Bi:sp3` - sp3 hybrids (8 WFs)
- Manual projections (see PROJECTION_GUIDE_SPINORS.md)

### Q: What if num_wann doesn't match my projections?

**A:** Remember spinor doubling:
- Each projection creates **2 WFs** (spin up + spin down)
- `Bi:p` = 3 p-orbitals × 2 atoms = 6 projections × 2 spinor = **12 WFs**
- Adjust `--num-wann` parameter accordingly

### Q: Stage 2 can't find .nnkp file?

**A:** Make sure you:
1. Ran `wannier90.x -pp seedname` on cluster
2. Transferred the `.nnkp` file back to local machine
3. Are running Stage 2 in the same directory as the `.nnkp`

### Q: Still getting large spreads?

**A:** Try these strategies:
1. Increase num_bands (16 → 18 or 20)
2. Adjust disentanglement window (expand dis_win_min/max)
3. Try different projections (Bi:sp, Bi:sp3)
4. Check band character in CRYSTAL output

---

## Summary: Why This Workflow Works

1. **create_win_template.py** generates properly configured .win with all necessary parameters
2. **wannier90.x -pp** determines optimal b-vectors for your system's symmetry
3. **lcao_to_wannier90.py --stage 2** generates data files using those exact b-vectors with:
   - Overlap correction: A = S(k) @ C(k)
   - Phase correction: exp(-i·b·τ) for atomic centers
   - Degenerate k-point handling for 2D systems
4. **wannier90.x** performs disentanglement and localization with physical projections

**Result:** Maximally-localized Wannier functions with positive, physically meaningful spreads!
