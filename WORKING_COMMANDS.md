# Working Command Sequence - Tested and Ready

## Complete Two-Stage Workflow

These commands are tested and working as of the latest fixes.

---

## Stage 1: Generate .win File

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

**Output:** `bismuth_improved.win`

**Note:** The script may detect this as non-SOC if the CRYSTAL file doesn't have explicit SOC flags. That's okay - we'll fix it in the next step.

---

## Edit .win File

You MUST edit `bismuth_improved.win` to:
1. Add SOC flag (`spinors = .true.`)
2. Set correct num_wann/num_bands
3. Add disentanglement parameters
4. Increase iterations

**Replace these lines:**

```fortran
num_wann = 3
num_bands = 3
```

**With:**

```fortran
num_wann = 12
num_bands = 16

! Spin-orbit coupling
spinors = .true.
```

**Add after num_bands:**

```fortran
! Disentanglement
dis_win_min = -8.5
dis_win_max = 0.5
dis_froz_min = -8.0
dis_froz_max = -1.1
dis_num_iter = 1000
dis_mix_ratio = 0.5
```

**Change iterations:**

```fortran
num_iter = 5000
conv_window = 5
```

**Final .win should have:**

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
```

---

## Preprocessing (on cluster)

Transfer `.win` and run Wannier90 preprocessor:

```bash
# Transfer
scp bismuth_improved.win cluster:~/wannier_test/

# On cluster
ssh cluster
cd wannier_test
wannier90.x -pp bismuth_improved
```

**Output:** `bismuth_improved.nnkp`

This generates the neighbor list that tells us which k-point pairs to compute overlaps for.

---

## Transfer .nnkp Back

```bash
scp cluster:~/wannier_test/bismuth_improved.nnkp ./
```

---

## Stage 2: Generate Data Files

Now generate `.eig`, `.amn`, `.mmn` using the `.nnkp`:

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
- `bismuth_improved.mmn` - Overlap matrices (phase-corrected + degenerate k-point handling)

**What happens:**
- Reads `.nnkp` neighbor list
- Parses atomic positions and creates basis-to-atom map
- Doubles basis_atom_map for SOC (112 orbitals from 56 basis functions)
- Solves eigenvalue problems for all 225 k-points
- Writes overlap-corrected AMN
- Writes phase-corrected MMN with degenerate k-point handling

---

## Transfer to Cluster and Run

```bash
# Transfer all files
scp bismuth_improved.{win,eig,amn,mmn,nnkp} cluster:~/wannier_test/

# Run on cluster
ssh cluster "cd wannier_test && wannier90.x bismuth_improved"
```

---

## Check Results

```bash
# Check final spreads
ssh cluster "cd wannier_test && grep 'Final Spread' bismuth_improved.wout"

# Check convergence
ssh cluster "cd wannier_test && grep 'CONV' bismuth_improved.wout | tail -20"

# Check Omega breakdown
ssh cluster "cd wannier_test && grep 'Omega' bismuth_improved.wout | grep 'SPRD' | tail -5"
```

**Expected:**

```
Omega I  =    3.57 Ang²      (gauge-invariant)
Omega D  =   30-80 Ang²      (diagonal, should decrease)
Omega OD =  <100 Ang²        (off-diagonal, KEY METRIC!)
----------------------------------------------------
Omega Total = <200 Ang²      (down from 718!)
```

---

## Key Points

✅ **Stage 1** creates `.win` → edit it manually for proper settings
✅ **Preprocessing** (`wannier90.x -pp`) generates `.nnkp` with neighbors
✅ **Stage 2** uses `.nnkp` to generate data files with correct overlaps
✅ **Result**: Phase-corrected MMN with overlap-corrected AMN

**Why this workflow?**
- Wannier90 determines optimal b-vector shells in preprocessing
- Stage 2 uses those exact neighbors for MMN calculation
- Phase correction requires atomic positions (parsed automatically)
- Degenerate k-point handling prevents `**********` overflow

---

## Troubleshooting

**Q: Stage 2 can't find .nnkp?**
→ Make sure you transferred it back after preprocessing!

**Q: Still getting large spreads?**
→ Try different projections: `Bi:sp`, `Bi:s`, or manual projections

**Q: "AttributeError: recip_lattice"?**
→ This is fixed in the latest version (commit after this document)

**Q: Non-SOC detected but it's SOC?**
→ Manual edit of .win is required (add `spinors = .true.`)

---

## Success Criteria

✅ **Excellent**: Omega_OD < 50 Ang², individual spreads < 15 Ang²
✅ **Good**: Omega_OD < 100 Ang², individual spreads < 20 Ang²
✅ **Acceptable**: Omega_OD < 200 Ang², individual spreads < 40 Ang²
❌ **Poor**: Omega_OD > 300 Ang² (try different projections)
