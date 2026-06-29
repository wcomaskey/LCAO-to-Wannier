#!/bin/bash
# Test Sc on a denser Wannier k-mesh (override --k-grid), full pipeline, check Omega_I.
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
W90=/home/wcom/wannier90/wannier90.x
IN=$HOME/sc_wannier_test/Sc.out
KX="${1:-12}"; KY="${2:-12}"; KZ="${3:-1}"
SEED="Sc${KX}x${KY}x${KZ}"
WORK=$HOME/sc_kgrid_test; rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || exit 1

echo "######## Stage 1  k-grid ${KX} ${KY} ${KZ} ########"
/usr/bin/time -v python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname "$SEED" \
    --spin alpha --memory low --k-grid "$KX" "$KY" "$KZ" 2>&1 \
    | grep -E 'frozen-window|num_wann|num_bands|STATUS|Created|Pruned|Overriding k-grid|Number of k-points|Maximum resident'
echo "######## wannier90 -pp ########"
$W90 -pp "$SEED" 2>&1 | grep -iE 'Error|Exiting'; echo "pp done"
grep -A1 'begin nnkpts' "$SEED.nnkp" | tail -1 | sed 's/^/nntot: /'
echo "######## Stage 2 ########"
python3 "$REPO/lcao_to_wannier90.py" --stage 2 --input "$IN" --seedname "$SEED" \
    --spin alpha --memory low --k-grid "$KX" "$KY" "$KZ" 2>&1 \
    | grep -E '\.eig:|\.amn:|\.mmn:|STATUS'
echo "######## wannier90 ########"
$W90 "$SEED" 2>&1 | tail -2
echo "######## RESULT (Omega_I should be POSITIVE) ########"
grep -iE 'Exiting|More states|Maximum number of disentang|All done' "$SEED.wout" | head -3
grep -E 'Omega I +=' "$SEED.wout" | tail -2
grep -E 'Sum of centres and spreads' "$SEED.wout" | tail -1
ls -la "${SEED}_hr.dat" 2>&1 | head -1
