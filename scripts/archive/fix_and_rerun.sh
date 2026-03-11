#!/bin/bash
# Fix workflow: Re-run with correct energy windows matching bismuth.win
# Answer to question: num_bands CANNOT be changed after creating .eig/.amn/.mmn files.
# These files are structured with exactly num_bands bands per k-point.
# You must regenerate them if num_bands changes.

set -e
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=== Re-running Workflow with Correct Energy Windows ==="
echo
echo "Current bismuth.win configuration:"
echo "  num_wann = 12"
echo "  num_bands = 12"
echo "  Energy window: [-12.0, 6.0] eV"
echo
echo "The issue: Previous Stage 2 run used wrong window (default [-5,3] eV)"
echo "The fix: Re-run Stage 2 with correct window matching .win file"
echo

# Step 1: Re-run preprocessing (required after .win was auto-updated)
echo "Step 1: Re-running Wannier90 preprocessing..."
echo "Command: ./external/wannier90-3.1.0/wannier90.x -pp bismuth"
./external/wannier90-3.1.0/wannier90.x -pp bismuth
echo "✓ Generated bismuth.nnkp"
echo

# Step 2: Run Stage 2 with CORRECT energy window matching .win file
echo "Step 2: Running Stage 2 with energy window [-12.0, 6.0] eV..."
echo "Command: python3 lcao_to_wannier90.py --stage 2 --input tests/Bismuth_basis_40.out --seedname bismuth --window -12.0 6.0"
echo
python3 lcao_to_wannier90.py --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -12.0 6.0
echo

echo "=== Ready for Final Wannier90 Run ==="
echo
echo "Files regenerated with correct band selection:"
ls -lh bismuth.{eig,amn,mmn} 2>/dev/null || echo "  (files will be listed above)"
echo
echo "Next step: Run Wannier90 to get well-localized Wannier functions"
echo "Command: wannier90.x bismuth"
echo
echo "Expected result: Much better localization (Omega_OD < 100 Ang²)"
