#!/bin/bash
# Quick installation script for LCAO-to-Wannier90 package

echo "========================================="
echo "LCAO-to-Wannier90 Quick Install"
echo "========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Creating virtual environment..."
python3 -m venv venv

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Installing dependencies..."
pip install --upgrade pip
pip install numpy scipy

echo "Step 4: Installing package in development mode..."
pip install -e .

echo ""
echo "========================================="
echo "✓ Installation Complete!"
echo "========================================="
echo ""
echo "To activate the environment in the future:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo ""
echo "Quick test commands:"
echo "  # Suggest energy window"
echo "  python3 tests/test_bismuth_full_workflow.py --mode suggest --target-wann 20"
echo ""
echo "  # Run with energy window"
echo "  python3 tests/test_bismuth_full_workflow.py --mode window --window -5.0 3.0"
echo ""
echo "See TESTING_GUIDE.md for complete documentation."
echo ""
