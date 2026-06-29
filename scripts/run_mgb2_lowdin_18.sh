#!/bin/bash
# Fine-grid (18x18x18) Lowdin confirmation: does a Lowdin MMN CONVERGE to the
# correct answer (matching the midpoint 18^3 reference: Omega_I +2.54, RMS 3.8meV)?
# If lowdin_no_berry -> ~3.8 meV it is correct (just bounded); if it plateaus
# higher it is a bounded-but-approximate method.
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
W90=/home/wcom/wannier90/wannier90.x
IN=$REPO/calculations/MgB2_basis_121.out
SEED=MgB2
BASEARGS="--spin alpha --method pdwf --k-grid 18 18 18"

for METHOD in lowdin lowdin_no_berry; do
  echo "################ 18^3 MMN method: $METHOD ################"
  WORK=$HOME/mgb2_${METHOD}_18; rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || exit 1
  python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname "$SEED" $BASEARGS --bands-plot 2>&1 \
      | grep -iE 'num_wann:|Created:' | head -2
  $W90 -pp "$SEED" 2>&1 | grep -iE 'Exiting|Error'
  python3 "$REPO/lcao_to_wannier90.py" --stage 2 --input "$IN" --seedname "$SEED" $BASEARGS \
      --mmn-method "$METHOD" 2>&1 | grep -iE '\.mmn:'
  $W90 "$SEED" 2>&1 | grep -iE 'Exiting|Error' | head -2
  grep -E 'Omega I +=|Omega Total' "$SEED.wout" | tail -2
  python3 "$REPO/scripts/plot_band_comparison.py" --input "$IN" --seedname "$SEED" \
      --spin alpha --ylim -16 12 -o "${SEED}_${METHOD}_18.png" 2>&1 | grep -iE 'RMS'
  cp "${SEED}_${METHOD}_18.png" "/mnt/c/Users/willi/AppData/Local/Temp/claude/C--Users-willi-OneDrive-Desktop-LCAO-to-Wannier/c42bf54e-fde7-4c6c-bfca-046026b0fad2/scratchpad/${SEED}_${METHOD}_18.png" 2>/dev/null
done
echo "=== DONE $(date) ==="
