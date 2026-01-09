#!/bin/bash
set -e

echo "=== Testing Complete LCAO-to-Wannier90 Workflow with SOC Fix ==="
echo

# Activate environment
source venv/bin/activate

# Clean old files
rm -f bismuth_final.*

# Stage 1: Generate .win
echo "Step 1: Generating .win file..."
python3 create_win_template.py --input tests/Bismuth_basis_40.out --seedname bismuth_final
echo

# Preprocessing
echo "Step 2: Running Wannier90 preprocessing..."
export PATH="/Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier/external/wannier90-3.1.0:$PATH"
wannier90.x -pp bismuth_final
echo

# Stage 2: Generate data files
echo "Step 3: Generating data files..."
timeout 120 python3 lcao_to_wannier90.py --stage 2 --input tests/Bismuth_basis_40.out --seedname bismuth_final --window -25.0 10.0 2>&1 | grep -E "Spin-orbit|Doubling|Created|Selected|bands"
echo

echo "=== Test Complete ==="
echo "Files generated:"
ls -lh bismuth_final.{win,nnkp,eig,amn,mmn} 2>/dev/null || echo "Some files missing"
