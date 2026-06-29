#!/bin/bash
# Runner for Sc Wannier pipeline test (executed inside WSL)
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
WORK="$HOME/sc_wannier_test"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

STAGE="${1:-1}"
cd "$WORK" || exit 1
echo "=== START stage $STAGE $(date) ==="
echo "REPO=$REPO"
echo "PYTHONPATH=$PYTHONPATH"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS  nproc=$(nproc)"
echo "python: $(which python3)"

/usr/bin/time -v python3 "$REPO/lcao_to_wannier90.py" \
    --stage "$STAGE" --input Sc.out --seedname Sc
RC=$?
echo "=== EXIT_CODE=$RC $(date) ==="
