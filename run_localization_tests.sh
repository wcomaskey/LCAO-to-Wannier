#!/bin/bash
#
# Run both localization test paths and compare results
#
# Usage:
#   ./run_localization_tests.sh [path1|path2|both]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Determine which path to run
PATH_TO_RUN=${1:-both}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p test_output

# Run Path 1: Automatic projections with disentanglement
if [ "$PATH_TO_RUN" = "path1" ] || [ "$PATH_TO_RUN" = "both" ]; then
    print_header "Running Path 1: Automatic Projections"

    if python3 test_localization_path1.py; then
        print_success "Path 1 setup complete"

        echo ""
        echo "Files generated:"
        ls -lh test_output/bismuth_path1_auto.* 2>/dev/null || echo "No files found"

        echo ""
        print_warning "Path 1 ready for cluster testing"
        echo "  Copy to cluster: scp test_output/bismuth_path1_auto.* cluster:~/path1/"
        echo "  Run: wannier90.x bismuth_path1_auto"
    else
        print_error "Path 1 setup failed"
        exit 1
    fi

    echo ""
fi

# Run Path 2: Manual projections
if [ "$PATH_TO_RUN" = "path2" ] || [ "$PATH_TO_RUN" = "both" ]; then
    print_header "Running Path 2: Manual Projections"

    if python3 test_localization_path2.py; then
        print_success "Path 2 setup complete"

        echo ""
        echo "Files generated:"
        ls -lh test_output/bismuth_path2_manual.* 2>/dev/null || echo "No files found"

        echo ""
        print_warning "Path 2 ready for cluster testing"
        echo "  Copy to cluster: scp test_output/bismuth_path2_manual.* cluster:~/path2/"
        echo "  Run: wannier90.x bismuth_path2_manual"
    else
        print_error "Path 2 setup failed"
        exit 1
    fi

    echo ""
fi

# Summary
if [ "$PATH_TO_RUN" = "both" ]; then
    print_header "Test Summary"

    echo ""
    echo "Two test configurations created:"
    echo ""
    echo "Path 1 (Recommended):"
    echo "  - 12 WFs, 16 DFT bands"
    echo "  - Automatic Bi:p projections"
    echo "  - Disentanglement ENABLED"
    echo "  - Easier, more robust"
    echo ""
    echo "Path 2 (Manual):"
    echo "  - 10 WFs, 10 DFT bands"
    echo "  - Manual projections (5 orbitals)"
    echo "  - No disentanglement"
    echo "  - Requires correct orbital selection"
    echo ""
    echo "Next steps:"
    echo "  1. Transfer both to cluster"
    echo "  2. Run both tests"
    echo "  3. Compare final spreads:"
    echo "     - Look for Omega_OD < 100 Ang²"
    echo "     - Individual spreads < 20 Ang²"
    echo "  4. Use whichever converges better"
    echo ""
fi

print_success "All tests ready for cluster execution!"
