#!/bin/bash
# wannier90.x needs Intel MKL + compiler runtime on the loader path.
set -u
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
SEED="${1:-Sc_lowmem}"
cd "$HOME/sc_wannier_test" || exit 1
echo "=== wannier90.x -pp $SEED ==="
/home/wcom/wannier90/wannier90.x -pp "$SEED"
echo "pp exit=$?"
ls -la "$SEED.nnkp" 2>&1
echo "--- neighbors per k (nntot) ---"
grep -A1 'begin nnkpts' "$SEED.nnkp" 2>/dev/null | tail -1
