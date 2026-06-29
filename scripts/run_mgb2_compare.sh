#!/bin/bash
# MgB2 (alpha) PDWF + spread-window-assist full pipeline + DFT/Wannier band comparison.
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
W90=/home/wcom/wannier90/wannier90.x
IN=$REPO/calculations/MgB2_basis_121.out
SEED=MgB2
ARGS="--spin alpha --method pdwf --spread-window-assist --k-grid 8 8 8"
WORK=$HOME/mgb2_compare; rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || exit 1

echo "######## Stage 1 $(date) ########"
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname "$SEED" $ARGS --bands-plot 2>&1 \
    | grep -iE 'dis_froz:|dis_win:|num_wann:|STATUS|Created:'
echo "######## -pp ########"
$W90 -pp "$SEED" 2>&1 | grep -iE 'Exiting|Error'
echo "######## Stage 2 ########"
python3 "$REPO/lcao_to_wannier90.py" --stage 2 --input "$IN" --seedname "$SEED" $ARGS 2>&1 \
    | grep -iE '\.eig:|\.amn:|\.mmn:|STATUS'
echo "######## wannier90 ########"
$W90 "$SEED" 2>&1 | tail -2
grep -E 'Omega I +=|Omega Total' "$SEED.wout" | tail -2
echo "######## band comparison plot ########"
python3 "$REPO/scripts/plot_band_comparison.py" --input "$IN" --seedname "$SEED" \
    --spin alpha -o "${SEED}_comparison.png"
echo "######## outputs ########"
ls -la "${SEED}_comparison.png" "${SEED}_band.dat" "${SEED}_hr.dat" 2>&1
echo "=== DONE $(date) ==="
