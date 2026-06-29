#!/bin/bash
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
cd "$HOME/sc_wannier_test" || exit 1
echo "=== Sc Stage 1 --spin both  $(date) ==="
/usr/bin/time -v python3 "$REPO/lcao_to_wannier90.py" \
    --stage 1 --input Sc.out --seedname Sc --spin both --memory low
echo "=== EXIT=$? $(date) ==="
echo "--- outputs ---"; ls -la Sc_alpha.win Sc_beta.win 2>&1
echo "--- Sc_alpha.win == original Sc.win (alpha-by-default)? ---"
diff <(grep -vE 'seedname|Generated' Sc.win) <(grep -vE 'seedname|Generated' Sc_alpha.win) \
    && echo "ALPHA MATCHES ORIGINAL DEFAULT" || echo "ALPHA DIFFERS FROM ORIGINAL"
echo "--- alpha vs beta differ? ---"
diff <(grep -vE 'seedname|Generated' Sc_alpha.win) <(grep -vE 'seedname|Generated' Sc_beta.win) >/dev/null \
    && echo "ALPHA == BETA (unexpected)" || echo "ALPHA != BETA (expected for spin-polarized)"
