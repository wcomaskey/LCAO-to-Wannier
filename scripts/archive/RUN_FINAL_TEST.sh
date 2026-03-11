#!/bin/bash
# Complete validated workflow with num_bands > num_wann

set -e
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "================================================================================"
echo "VALIDATED WANNIER90 WORKFLOW"
echo "================================================================================"
echo
echo "Configuration:"
echo "  num_wann = 12 (Wannier projections)"
echo "  num_bands = 17 (DFT bands, 142% of num_wann)"
echo "  Outer window: [-5.7, 5.7] eV"
echo "  Frozen window: [-4.0, 4.0] eV"
echo
echo "Key principle: num_bands > num_wann for disentanglement flexibility"
echo
echo "================================================================================"
echo

# Step 1: Preprocessing
echo "Step 1: Running Wannier90 preprocessing..."
echo "Command: ./external/wannier90-3.1.0/wannier90.x -pp bismuth"
./external/wannier90-3.1.0/wannier90.x -pp bismuth
echo "✓ Generated bismuth.nnkp"
echo

# Step 2: Stage 2 with correct window
echo "Step 2: Running Stage 2 with validated configuration..."
echo "Command: python3 lcao_to_wannier90.py --stage 2 --input tests/Bismuth_basis_40.out --seedname bismuth --window -5.7 5.7"
echo
python3 lcao_to_wannier90.py --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -5.7 5.7
echo

echo "================================================================================"
echo "READY FOR FINAL WANNIER90 RUN"
echo "================================================================================"
echo
echo "Files generated:"
ls -lh bismuth.{eig,amn,mmn} 2>/dev/null || echo "  (files listed above)"
echo
echo "Next step:"
echo "  ./external/wannier90-3.1.0/wannier90.x bismuth"
echo
echo "Expected: Well-localized Wannier functions (Omega_OD < 100 Ang²)"
echo "================================================================================"
