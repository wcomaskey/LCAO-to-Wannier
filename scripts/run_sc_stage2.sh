#!/bin/bash
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
cd "$HOME/sc_wannier_test" || exit 1
echo "=== Sc Stage 2 --memory low  $(date) ==="
/usr/bin/time -v python3 "$REPO/lcao_to_wannier90.py" \
    --stage 2 --input Sc.out --seedname Sc_lowmem --memory low
echo "=== EXIT=$? $(date) ==="
ls -la Sc_lowmem.eig Sc_lowmem.amn Sc_lowmem.mmn 2>&1
