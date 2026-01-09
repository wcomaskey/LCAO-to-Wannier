# Pre-Flight Checklist

Use this checklist before running the workflow to ensure everything is ready.

---

## Initial Setup

- [ ] Virtual environment activated
  ```bash
  source venv/bin/activate
  ```

- [ ] Input file exists and is accessible
  ```bash
  ls tests/Bismuth_basis_40.out  # Or your input file
  ```

- [ ] Scripts are executable
  ```bash
  chmod +x run_full_workflow.sh
  ```

---

## Stage 1: .win Generation

- [ ] Run create_win_template.py
  ```bash
  python3 create_win_template.py --input tests/Bismuth_basis_40.out --seedname YOUR_NAME
  ```

- [ ] Check .win file was created
  ```bash
  ls -lh YOUR_NAME.win
  ```

- [ ] Verify key parameters in .win
  ```bash
  grep -E "num_wann|num_bands|spinors" YOUR_NAME.win
  ```

  Should see:
  ```
  num_wann = 12
  num_bands = 16
  spinors = .true.
  ```

- [ ] Verify atomic positions exist
  ```bash
  grep -A 3 "begin atoms_frac" YOUR_NAME.win
  ```

  Should see 2 Bi atoms with fractional coordinates

- [ ] Verify projections are set
  ```bash
  grep -A 2 "begin projections" YOUR_NAME.win
  ```

  Should see: `Bi:p`

---

## Preprocessing

**Option A: On Local Machine (if Wannier90 installed)**

- [ ] Set PATH to Wannier90
  ```bash
  export PATH="external/wannier90-3.1.0:$PATH"
  ```

- [ ] Run preprocessing
  ```bash
  wannier90.x -pp YOUR_NAME
  ```

- [ ] Check .nnkp was created
  ```bash
  ls -lh YOUR_NAME.nnkp
  ```

**Option B: On Cluster**

- [ ] Transfer .win to cluster
  ```bash
  scp YOUR_NAME.win cluster:~/wannier_test/
  ```

- [ ] Run preprocessing on cluster
  ```bash
  ssh cluster "cd wannier_test && wannier90.x -pp YOUR_NAME"
  ```

- [ ] Transfer .nnkp back
  ```bash
  scp cluster:~/wannier_test/YOUR_NAME.nnkp ./
  ```

---

## Stage 2: Data File Generation

- [ ] Verify .nnkp exists locally
  ```bash
  ls -lh YOUR_NAME.nnkp
  ```

- [ ] Run Stage 2
  ```bash
  python3 lcao_to_wannier90.py \
      --stage 2 \
      --input tests/Bismuth_basis_40.out \
      --seedname YOUR_NAME \
      --window -8.5 0.5
  ```

- [ ] Check all data files were created
  ```bash
  ls -lh YOUR_NAME.{eig,amn,mmn}
  ```

- [ ] Verify file sizes are reasonable
  ```
  .eig:  ~20 KB   (eigenvalues)
  .amn:  ~111 KB  (projections)
  .mmn:  ~654 KB  (overlaps)
  ```

- [ ] Check AMN header
  ```bash
  head -1 YOUR_NAME.amn
  ```

  Should say: "Created by LCAO-to-Wannier90 (Overlap-Corrected)"

- [ ] Check MMN header
  ```bash
  head -1 YOUR_NAME.mmn
  ```

  Should say: "Created by LCAO-to-Wannier90 (Phase-Corrected)"

---

## Final Wannier90 Run

**On Cluster:**

- [ ] Transfer all files to cluster
  ```bash
  scp YOUR_NAME.{win,nnkp,eig,amn,mmn} cluster:~/wannier_test/
  ```

- [ ] Run Wannier90
  ```bash
  ssh cluster "cd wannier_test && wannier90.x YOUR_NAME"
  ```

- [ ] Check for errors
  ```bash
  ssh cluster "cd wannier_test && grep -i error YOUR_NAME.wout"
  ```

  Should be empty or only warnings

- [ ] Check convergence
  ```bash
  ssh cluster "cd wannier_test && grep CONV YOUR_NAME.wout | tail -5"
  ```

  Should see "CONV" indicating convergence

---

## Results Validation

- [ ] Check final spreads
  ```bash
  ssh cluster "cd wannier_test && grep 'Final Spread' YOUR_NAME.wout"
  ```

- [ ] All spreads are positive (no negative values)

- [ ] No `**********` overflow in output

- [ ] Check Omega breakdown
  ```bash
  ssh cluster "cd wannier_test && grep 'Omega' YOUR_NAME.wout | grep SPRD | tail -5"
  ```

- [ ] Verify localization quality:
  - [ ] Omega_OD < 100 Ang² (excellent)
  - [ ] Individual spreads < 20 Ang² (well-localized)
  - [ ] Total spread < 200 Ang²

- [ ] Check individual WF centers
  ```bash
  ssh cluster "cd wannier_test && grep 'WF centre and spread' YOUR_NAME.wout"
  ```

---

## Success Criteria

✅ All checks above passed
✅ Wannier90 converged (CONV appears)
✅ All spreads are positive
✅ Omega_OD < 100 Ang² (excellent localization)
✅ No errors in .wout file

---

## If Something Goes Wrong

### .win generation failed
- Check input file path
- Check that lcao_wannier is installed
- Review error message

### Preprocessing failed
- Check .win file syntax
- Verify lattice vectors are valid
- Check k-point mesh matches input

### Stage 2 failed
- Verify .nnkp exists
- Check input file path
- Verify energy window is reasonable
- Check error message in output

### Wannier90 failed
- Check all 5 files transferred to cluster
- Check .wout file for specific error
- Verify num_wann matches projections
- Check disentanglement windows

### Poor localization (Omega_OD > 300 Ang²)
- Try different projections (Bi:sp, Bi:sp3)
- Increase num_bands (+4 to +8)
- Adjust disentanglement windows
- Check band character in DFT output

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "Could not open .nnkp" | Run preprocessing first |
| "num_wann mismatch" | Check spinor doubling (each proj → 2 WFs) |
| Negative spreads | Update to latest code version |
| `**********` overflow | Update to latest code version |
| "AttributeError: recip_lattice" | Update to latest code version |
| Poor localization | Try different projections |
| Won't converge | Increase num_iter to 10000 |

---

## Documentation References

- **Quick commands:** QUICK_REFERENCE.md
- **Detailed workflow:** TWO_STAGE_WORKFLOW.md
- **Technical details:** SOLUTION_SUMMARY.md
- **Projection setup:** PROJECTION_GUIDE_SPINORS.md
- **Understanding results:** LOCALIZATION_ANALYSIS.md

---

**Save this checklist and use it each time you run the workflow!**
