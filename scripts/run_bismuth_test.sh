#!/bin/bash
# Isolate grid-density vs SOC: Bismuth (2D SOC) at native 15x15 (dense) and 3x3 (coarse).
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
W90=/home/wcom/wannier90/wannier90.x
IN=$REPO/tests/Bismuth_basis_40.out

run_one() {
    SEED="$1"; shift
    KG="$*"   # "" for native, or "--k-grid 3 3 1"
    WORK="$HOME/bi_test_${SEED}"
    rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || return 1
    echo "################ $SEED  (kgrid: ${KG:-native}) ################"
    python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname "$SEED" --memory low $KG 2>&1 \
        | grep -E 'frozen-window|num_wann|num_bands|STATUS|Overriding k-grid|Created'
    $W90 -pp "$SEED" >/dev/null 2>&1; echo "pp exit=$?"
    python3 "$REPO/lcao_to_wannier90.py" --stage 2 --input "$IN" --seedname "$SEED" --memory low $KG 2>&1 \
        | grep -E '\.eig:|\.amn:|\.mmn:|STATUS'
    $W90 "$SEED" >/dev/null 2>&1; echo "w90 exit=$?"
    echo "---------------- RESULT $SEED ----------------"
    grep -iE 'Exiting|More states|All done' "$SEED.wout" | head -2
    grep -E 'mp_grid' "$SEED.win" | head -1
    grep -E 'Omega I +=' "$SEED.wout" | tail -1
    grep -E 'Sum of centres and spreads' "$SEED.wout" | tail -1
    echo
}

run_one Bi_native
run_one Bi_3x3 --k-grid 3 3 1
echo "================ ALL DONE $(date) ================"
