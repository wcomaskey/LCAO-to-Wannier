#!/bin/bash
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
cd ~/sc_wannier_test || exit 1
echo "=== Sc Stage 1 --memory low  $(date) ==="
/usr/bin/time -v python3 "$REPO/lcao_to_wannier90.py" \
    --stage 1 --input Sc.out --seedname Sc_lowmem --memory low
echo "=== EXIT=$? $(date) ==="
