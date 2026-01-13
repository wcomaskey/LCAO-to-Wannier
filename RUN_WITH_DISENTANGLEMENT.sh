#!/bin/bash
# Run with proper disentanglement: num_bands (18) > num_wann (12)

set -e
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "================================================================================"
echo "WANNIER90 WITH DISENTANGLEMENT"
echo "================================================================================"
echo
echo "Configuration:"
echo "  num_wann = 12 (Wannier projections)"
echo "  num_bands = 18 (DFT bands - 150% of num_wann)"
echo "  Outer window: [-15.0, 10.0] eV (WIDE to contain 18+ bands)"
echo "  Frozen window: [-6.0, 4.0] eV"
echo
echo "This allows Wannier90 to DISENTANGLE the 12 Bi p-orbitals from 18 bands"
echo "================================================================================"
echo

# Step 1: Preprocessing
echo "Step 1: Running Wannier90 preprocessing..."
./external/wannier90-3.1.0/wannier90.x -pp bismuth
echo "✓ Generated bismuth.nnkp"
echo

# Step 2: Stage 2 with WIDE window
echo "Step 2: Running Stage 2 with WIDE energy window..."
echo "Command: python3 lcao_to_wannier90.py --stage 2 --input tests/Bismuth_basis_40.out --seedname bismuth --window -15.0 10.0"
echo
python3 lcao_to_wannier90.py --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -15.0 10.0

echo
echo "================================================================================"
echo "FILES READY FOR WANNIER90"
echo "================================================================================"
echo
ls -lh bismuth.{win,nnkp,eig,amn,mmn}
echo
echo "Verifying .eig file has 18 bands:"
echo "  Total lines should be: 18 bands × 225 k-points = 4050"
wc -l bismuth.eig
echo
echo "Next step: Run Wannier90.x on cluster"
echo "  Expected: Positive Omega_I, Omega_OD < 100 Ang²"
echo "================================================================================"
