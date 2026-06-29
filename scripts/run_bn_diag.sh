#!/bin/bash
# Diagnostic: BN Stage 1 (full output) to find why the pipeline failed.
REPO=/mnt/c/Users/willi/OneDrive/Desktop/LCAO-to-Wannier
export PYTHONPATH=$REPO OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
IN=$REPO/calculations/BN_1C.out
cd /tmp; rm -rf bn_mid; mkdir bn_mid; cd bn_mid || exit 1
echo "######## Stage 1 ########"
python3 "$REPO/lcao_to_wannier90.py" --stage 1 --input "$IN" --seedname BN \
    --method pdwf --k-grid 12 12 4 --bands-plot 2>&1 | tail -40
echo "######## files ########"
ls -la BN.* 2>&1
