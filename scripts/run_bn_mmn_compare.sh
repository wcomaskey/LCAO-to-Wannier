#!/bin/bash
# Non-centrosymmetric h-BN (P-6m2): discriminate MMN methods by WANNIER CENTRES.
# Guiding centres are DISABLED so the WF centres are free to settle where the MMN
# dictates (guiding would pull them to atoms and mask the Berry-phase shift).
# If lowdin and lowdin_no_berry give DIFFERENT centres, the Berry phase matters
# here (unlike MgB2); whichever matches the midpoint reference is correct.
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
W90=/home/wcom/wannier90/wannier90.x
IN=$REPO/calculations/BN_1C.out
GRID="${1:-12 12 4}"
BASE=/tmp/bn_cmp; rm -rf "$BASE"; mkdir -p "$BASE"

# Stage 1 once (shared .win), then disable guiding_centres for the centre test.
cd "$BASE" || exit 1
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname BN \
    --method pdwf --k-grid $GRID --bands-plot 2>&1 | grep -iE 'num_wann:|dis_froz|dis_win|Created:'
sed -i 's/^guiding_centres.*/guiding_centres = .false./' BN.win
echo "guiding_centres line now: $(grep -i guiding BN.win)"
cp BN.win BN_stage1.win

for METHOD in midpoint lowdin lowdin_no_berry; do
  echo "################ $METHOD (grid $GRID) ################"
  W=$BASE/$METHOD; rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1
  cp "$BASE"/BN_stage1.win BN.win
  cp "$IN" ./ 2>/dev/null
  $W90 -pp BN 2>&1 | grep -iE 'Exiting|Error'
  python3 "$REPO/lcao_to_wannier90.py" --stage 2 --input "$IN" --seedname BN \
      --method pdwf --k-grid $GRID --mmn-method "$METHOD" 2>&1 | grep -iE '\.mmn:|Exiting|Error' | head -2
  $W90 BN 2>&1 | grep -iE 'Exiting|Error' | head -2
  echo "--- Omega ($METHOD) ---"; grep -E 'Omega I +=|Omega D +=|Omega OD|Omega Total' BN.wout | tail -4
  echo "--- Final WF centres ($METHOD) ---"; grep -A18 'Final State' BN.wout | grep 'WF centre' | head -18
done
echo "=== DONE $(date) ==="
