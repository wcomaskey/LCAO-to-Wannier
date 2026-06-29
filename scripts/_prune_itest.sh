#!/bin/bash
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
IN=$REPO/tests/Bismuth_basis_40.out

python3 -m py_compile "$REPO/lcao_to_wannier90.py" "$REPO/lcao_wannier/utils.py" \
    && echo "COMPILE OK" || { echo "COMPILE FAIL"; exit 1; }

rm -rf /tmp/pr_on /tmp/pr_off; mkdir -p /tmp/pr_on /tmp/pr_off

cd /tmp/pr_on
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname Bi \
    --memory low > on.log 2>&1; echo "prune-on exit=$?"
grep -E "Pruned|Prepared .* R-vectors" on.log | head

cd /tmp/pr_off
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname Bi \
    --memory low --no-prune > off.log 2>&1; echo "no-prune exit=$?"
grep -E "Pruned|Prepared .* R-vectors" off.log | head

echo "### diff Bi.win (prune-on vs no-prune) ###"
if diff -q /tmp/pr_on/Bi.win /tmp/pr_off/Bi.win >/dev/null 2>&1; then
    echo "Bi.win IDENTICAL  ✓ (pruning is result-preserving)"
else
    echo "DIFFERS:"; diff /tmp/pr_on/Bi.win /tmp/pr_off/Bi.win | head
fi
