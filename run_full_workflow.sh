#!/bin/bash
# Complete LCAO-to-Wannier90 Workflow Script
# Usage: ./run_full_workflow.sh [seedname]

set -e  # Exit on error

SEEDNAME=${1:-bismuth_improved}
INPUT_FILE="tests/Bismuth_basis_40.out"
WINDOW_MIN=-8.5
WINDOW_MAX=0.5

echo "=========================================="
echo "LCAO-to-Wannier90 Full Workflow"
echo "=========================================="
echo "Seedname: $SEEDNAME"
echo "Input: $INPUT_FILE"
echo "Energy window: [$WINDOW_MIN, $WINDOW_MAX] eV"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Stage 1: Generate .win file
echo "=========================================="
echo "Stage 1: Generating .win template"
echo "=========================================="
python3 create_win_template.py \
    --input "$INPUT_FILE" \
    --seedname "$SEEDNAME" \
    --num-wann 12 \
    --num-bands 16

if [ ! -f "${SEEDNAME}.win" ]; then
    echo "ERROR: .win file not created!"
    exit 1
fi

echo ""
echo "✓ Created ${SEEDNAME}.win"
echo ""

# Check if .nnkp exists
if [ -f "${SEEDNAME}.nnkp" ]; then
    echo "=========================================="
    echo "Found existing ${SEEDNAME}.nnkp"
    echo "=========================================="
    echo ""
    echo "Stage 2: Generating data files using .nnkp"
    echo "=========================================="

    python3 lcao_to_wannier90.py \
        --stage 2 \
        --input "$INPUT_FILE" \
        --seedname "$SEEDNAME" \
        --window $WINDOW_MIN $WINDOW_MAX

    echo ""
    echo "=========================================="
    echo "✓ Workflow Complete!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    ls -lh ${SEEDNAME}.{win,eig,amn,mmn,nnkp} 2>/dev/null || true
    echo ""
    echo "Next steps:"
    echo "  1. Transfer all files to cluster:"
    echo "     scp ${SEEDNAME}.* cluster:~/wannier_test/"
    echo ""
    echo "  2. Run Wannier90 on cluster:"
    echo "     ssh cluster 'cd wannier_test && wannier90.x ${SEEDNAME}'"
    echo ""
    echo "  3. Check results:"
    echo "     ssh cluster 'cd wannier_test && grep \"Final Spread\" ${SEEDNAME}.wout'"

else
    echo "=========================================="
    echo "Preprocessing Required"
    echo "=========================================="
    echo ""
    echo "The .nnkp file does not exist yet."
    echo ""
    echo "Next steps:"
    echo "  1. Transfer .win file to cluster:"
    echo "     scp ${SEEDNAME}.win cluster:~/wannier_test/"
    echo ""
    echo "  2. Run Wannier90 preprocessing on cluster:"
    echo "     ssh cluster 'cd wannier_test && wannier90.x -pp ${SEEDNAME}'"
    echo ""
    echo "  3. Copy .nnkp file back:"
    echo "     scp cluster:~/wannier_test/${SEEDNAME}.nnkp ./"
    echo ""
    echo "  4. Run this script again to generate data files:"
    echo "     ./run_full_workflow.sh $SEEDNAME"
fi
