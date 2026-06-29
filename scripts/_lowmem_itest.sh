#!/bin/bash
# Integration test: Stage 1 with --memory fast vs --memory low must match.
set -u
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
IN=$REPO/tests/Bismuth_basis_40.out

rm -rf /tmp/lm_fast /tmp/lm_low
mkdir -p /tmp/lm_fast /tmp/lm_low

echo "### FAST ###"
cd /tmp/lm_fast
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname Bi --memory fast \
    > fast.log 2>&1; echo "fast exit=$?"
grep -E "Number of AOs|Spin-orbit|num_wann|num_bands|Final selected|Basis size" fast.log | head

echo "### LOW ###"
cd /tmp/lm_low
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname Bi --memory low \
    > low.log 2>&1; echo "low exit=$?"
grep -E "memory=low|Number of AOs|Spin-orbit|num_wann|num_bands|Final selected|Basis size" low.log | head

echo "### DIFF .win ###"
if diff -q /tmp/lm_fast/Bi.win /tmp/lm_low/Bi.win >/dev/null 2>&1; then
    echo "Bi.win IDENTICAL  ✓"
else
    echo "Bi.win DIFFERS:"; diff /tmp/lm_fast/Bi.win /tmp/lm_low/Bi.win | head -30
fi
echo "### peak RSS (KB) fast vs low ==="
grep -i "Maximum resident" /tmp/lm_fast/fast.log /tmp/lm_low/low.log 2>/dev/null || true
echo "### tail of low.log (any errors) ###"
tail -5 /tmp/lm_low/low.log
