#!/bin/bash
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
cd "$HOME/sc_wannier_test" || exit 1
echo "=== Sc Stage 2 --spin both  $(date) ==="
/usr/bin/time -v python3 "$REPO/lcao_to_wannier90.py" \
    --stage 2 --input Sc.out --seedname Sc --spin both --memory low
echo "=== EXIT=$? $(date) ==="
echo "--- per-channel data files ---"
ls -la Sc_alpha.eig Sc_alpha.amn Sc_alpha.mmn Sc_beta.eig Sc_beta.amn Sc_beta.mmn 2>&1
echo "--- Sc_alpha.eig == earlier alpha-only Sc_lowmem.eig? (same channel+grid) ---"
diff -q Sc_alpha.eig Sc_lowmem.eig >/dev/null 2>&1 && echo "ALPHA EIG MATCHES Sc_lowmem" || echo "alpha eig differs from Sc_lowmem"
echo "--- alpha vs beta eig differ? ---"
diff -q Sc_alpha.eig Sc_beta.eig >/dev/null 2>&1 && echo "alpha==beta eig (unexpected)" || echo "alpha!=beta eig (expected)"
