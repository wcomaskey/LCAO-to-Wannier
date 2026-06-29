#!/bin/bash
# Regression test: full pipeline on a known-good case, check Omega_I > 0.
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
W90=/home/wcom/wannier90/wannier90.x
IN=$REPO/calculations/MgB2_basis_121.out
SEED=MgB2
SPIN="${1:-alpha}"      # single channel matches pre-today behavior
EXTRA="${2:-}"          # e.g. --no-prune
WORK=$HOME/mgb2_test
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || exit 1

echo "######## Stage 1 (spin=$SPIN $EXTRA) ########"
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname "$SEED" --spin "$SPIN" $EXTRA 2>&1 \
    | grep -E 'frozen-window|num_wann|num_bands|STATUS|Created|Pruned|Unique R|Error|Traceback'
echo "######## wannier90 -pp ########"
$W90 -pp "$SEED" 2>&1 | grep -iE 'Error|Exiting' ; echo "pp exit=$?"
echo "######## Stage 2 (spin=$SPIN $EXTRA) ########"
python3 "$REPO/lcao_to_wannier90.py" --stage 2 --input "$IN" --seedname "$SEED" --spin "$SPIN" $EXTRA 2>&1 \
    | grep -E '\.eig:|\.amn:|\.mmn:|STATUS|Error|Traceback'
echo "######## wannier90 ########"
$W90 "$SEED" 2>&1 | tail -2
echo "######## RESULT ########"
grep -iE 'Exiting|More states|Maximum number of disentang' "$SEED.wout" | head -3
echo "--- Omega_I and final spreads ---"
grep -E 'Omega I +=|Omega_I' "$SEED.wout" | tail -3
grep -E 'Total Omega|Sum of centres and spreads' "$SEED.wout" | tail -2
ls -la "${SEED}_hr.dat" 2>&1 | head -1
