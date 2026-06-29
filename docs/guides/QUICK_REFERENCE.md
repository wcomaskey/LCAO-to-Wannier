# Quick Reference - LCAO to Wannier90

## One-Command Workflow

```bash
# Automated workflow (recommended)
./run_full_workflow.sh bismuth_improved
```

Follow the instructions printed by the script.

---

## Manual Workflow

### Local Machine

```bash
# Activate environment
source venv/bin/activate

# Stage 1: Generate .win
python3 create_win_template.py \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_improved

# Transfer to cluster
scp bismuth_improved.win cluster:~/wannier_test/
```

### On Cluster

```bash
# Preprocessing
cd wannier_test
wannier90.x -pp bismuth_improved
```

### Back on Local Machine

```bash
# Get .nnkp
scp cluster:~/wannier_test/bismuth_improved.nnkp ./

# Stage 2: Generate data files
python3 lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth_improved \
    --window -8.5 0.5

# Transfer all files
scp bismuth_improved.* cluster:~/wannier_test/
```

### Final Run on Cluster

```bash
# Run Wannier90
cd wannier_test
wannier90.x bismuth_improved

# Check results
grep "Final Spread" bismuth_improved.wout
grep "Omega" bismuth_improved.wout | grep SPRD | tail -5
```

---

## Common Variations

### Different Number of WFs

```bash
# 10 WFs (5 projections × 2 spinor)
python3 create_win_template.py \
    --input INPUT.out \
    --seedname NAME \
    --num-wann 10 \
    --num-bands 14

# Edit .win to use 5 manual projections instead of Bi:p
```

### Different Projections

Edit the `.win` file before preprocessing:

```fortran
! s-orbitals only (4 WFs)
begin projections
Bi:s
end projections

! s+p mixed (16 WFs)
begin projections
Bi:sp
end projections

! sp3 hybrids (16 WFs)
begin projections
Bi:sp3
end projections
```

### Different Energy Window

```bash
python3 lcao_to_wannier90.py \
    --stage 2 \
    --input INPUT.out \
    --seedname NAME \
    --window -10.0 2.0  # Wider window
```

---

## Memory, Spin & Validation Options

### Spin-polarized (UNRESTRICTED) outputs

```bash
# Both spin channels -> NAME_alpha.* and NAME_beta.* (default for spin-polarized)
python3 lcao_to_wannier90.py --stage 1 --input INPUT.out --seedname NAME --spin both
wannier90.x -pp NAME_alpha   # run -pp per channel
wannier90.x -pp NAME_beta
python3 lcao_to_wannier90.py --stage 2 --input INPUT.out --seedname NAME --spin both

# A single channel (uses NAME as-is)
python3 lcao_to_wannier90.py --stage 1 --input INPUT.out --seedname NAME --spin alpha
```
`--spin` errors on restricted or two-component SOC outputs (it does not apply).

### Large inputs (low memory)

```bash
# Predict peak RAM first
python3 scripts/estimate_memory.py INPUT.out --available 26

# Streaming, low-memory parse (~2x lower peak; default prunes all-zero R-vectors)
python3 lcao_to_wannier90.py --stage 1 --input INPUT.out --seedname NAME --memory low
#   --no-prune              keep all-zero R-vectors
#   --prune-threshold 1e-10 also drop negligible cells
```

### Validate a .win before running wannier90

```bash
# Stage 1/2 already print a PASS/FAIL disentanglement check automatically.
# To re-check an existing seedname (.win + .eig):
python3 scripts/check_win.py NAME
```
See `DISENTANGLEMENT_WINDOW_RULES.md` for what is checked and the frozen-window
rule.

---

## File Locations

### Generated Files

```
bismuth_improved.win    # Wannier90 input
bismuth_improved.nnkp   # Neighbor list (from preprocessing)
bismuth_improved.eig    # Eigenvalues
bismuth_improved.amn    # Projection matrices
bismuth_improved.mmn    # Overlap matrices
bismuth_improved.wout   # Wannier90 output
bismuth_improved_hr.dat # Hamiltonian in WF basis
```

### Scripts

```
create_win_template.py      # Stage 1: Generate .win
lcao_to_wannier90.py        # Stage 2: Generate data files
run_full_workflow.sh        # Automated workflow
```

### Documentation

```
TWO_STAGE_WORKFLOW.md       # Complete step-by-step guide
SOLUTION_SUMMARY.md         # Technical summary
QUICK_REFERENCE.md          # This file
PROJECTION_GUIDE_SPINORS.md # Projection setup guide
LOCALIZATION_ANALYSIS.md    # Technical analysis
```

---

## Checking Results

### Quick Check

```bash
# On cluster
grep "Final Spread" *.wout
```

Good result: `<20 Ang²` per WF

### Detailed Check

```bash
# Omega breakdown
grep "Omega" *.wout | grep SPRD | tail -5

# Convergence
grep "CONV" *.wout | tail -20

# Individual WF spreads
grep "WF centre and spread" *.wout
```

### Success Criteria

- ✅ All spreads positive
- ✅ Omega_OD < 100 Ang²
- ✅ Individual spreads < 20 Ang²
- ✅ "CONV" appears (converged)

---

## Troubleshooting

### "Could not open .nnkp file"

**Solution:** Run preprocessing first:
```bash
wannier90.x -pp seedname
```

### "num_wann mismatch"

**Solution:** Check spinor doubling:
- Each projection → 2 WFs (spin up + down)
- Bi:p → 6 projections → 12 WFs

### Still large spreads (>100 Ang²)

**Try:**
1. Different projections (edit .win)
2. Increase num_bands (+4 to +8)
3. Adjust disentanglement window
4. Check band character in DFT output

### "AttributeError: recip_lattice"

**Solution:** Update to latest code version:
```bash
git pull
```

---

## Parameter Guidelines

### num_wann
- Bismuth p-orbitals: 12 (3 × 2 atoms × 2 spinor)
- Bismuth s-orbitals: 4 (1 × 2 atoms × 2 spinor)
- Rule: Count projections, then double for spinors

### num_bands
- Typical: num_wann + 4 to 8
- More bands = better disentanglement
- Must be ≥ num_wann

### Energy Window
- Outer window: Include all relevant bands
- Frozen window: Target states only
- Check Fermi energy in CRYSTAL output

### Iterations
- Standard: 5000 (usually sufficient)
- If not converged: increase to 10000

---

## Quick Tips

1. **Always use automated script first**
   ```bash
   ./run_full_workflow.sh NAME
   ```

2. **Check .win before preprocessing**
   - Verify num_wann matches projections
   - Verify atomic positions look reasonable
   - Verify k-point count matches DFT

3. **Save the .nnkp file**
   - It's system-specific
   - No need to regenerate if .win unchanged

4. **Don't skip preprocessing**
   - Stage 2 needs the .nnkp neighbor list
   - Wannier90 determines optimal b-vectors

5. **Convergence is key**
   - Check for "CONV" in output
   - If not converged, increase num_iter

---

## Version Info

- LCAO-to-Wannier90: Latest (2025-01-09)
- All fixes applied:
  - ✓ Overlap correction
  - ✓ Phase correction
  - ✓ Degenerate k-point handling
  - ✓ SOC support
  - ✓ Atom parsing fix

For detailed technical info, see SOLUTION_SUMMARY.md
