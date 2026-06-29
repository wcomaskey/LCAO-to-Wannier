#!/bin/bash
# Run the full wannier90.x wannierization sequentially for both spin channels.
set -u
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/2025.1/lib:${LD_LIBRARY_PATH:-}
cd "$HOME/sc_wannier_test" || exit 1
for SEED in Sc_alpha Sc_beta; do
    echo "############################################################"
    echo "### wannier90.x $SEED  $(date)"
    echo "############################################################"
    /home/wcom/wannier90/wannier90.x "$SEED"
    echo "exit=$?"
    echo "--- final spreads / convergence ($SEED.wout) ---"
    grep -E "Omega_|Final State|<-- DIS|CONV|converged|Maximum" "$SEED.wout" 2>/dev/null | tail -8
    echo "--- outputs ---"
    ls -la "${SEED}_hr.dat" "${SEED}_band.dat" "${SEED}.wout" 2>&1
done
echo "=== ALL DONE $(date) ==="
