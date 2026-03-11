#!/bin/bash
#
# Run projection/localization tests using the bismuth_full_workflow test
# with different projection configurations
#
# Usage:
#   ./run_projection_tests.sh [path1|path2|both]
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() { echo -e "${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }

PATH_TO_RUN=${1:-both}

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

mkdir -p test_output

# Path 1: 12 WFs with disentanglement, automatic Bi:p projections
if [ "$PATH_TO_RUN" = "path1" ] || [ "$PATH_TO_RUN" = "both" ]; then
    print_header "Path 1: 12 WFs + Disentanglement"

    echo "Generating files with automatic Bi:p projections..."
    python3 tests/test_bismuth_full_workflow.py \
        --mode window \
        --window -8.5 0.0 \
        --seedname bismuth_path1_auto \
        --output-dir test_output

    if [ -f "test_output/bismuth_path1_auto.win" ]; then
        print_success "Base files generated"

        # Modify .win file for disentanglement
        echo "Adding disentanglement parameters..."

        # Update num_bands and add disentanglement
        sed -i.bak 's/^num_bands = .*/num_bands = 16/' test_output/bismuth_path1_auto.win

        # Add disentanglement block after num_bands
        awk '/^num_bands = 16/{
            print
            print ""
            print "! Disentanglement windows"
            print "dis_win_min = -8.5"
            print "dis_win_max = 0.0"
            print "dis_froz_min = -8.0"
            print "dis_froz_max = -1.1"
            print "dis_num_iter = 1000"
            print "dis_mix_ratio = 0.5"
            next
        }1' test_output/bismuth_path1_auto.win > test_output/bismuth_path1_auto.win.tmp
        mv test_output/bismuth_path1_auto.win.tmp test_output/bismuth_path1_auto.win

        # Change projections to Bi:p and increase iterations
        sed -i.bak 's/random/Bi:p/' test_output/bismuth_path1_auto.win
        sed -i.bak 's/^num_iter = .*/num_iter = 5000/' test_output/bismuth_path1_auto.win

        print_success "Path 1 configured"
        echo ""
        echo "Configuration:"
        echo "  - num_wann = 12 (will be set by band analysis, target 12)"
        echo "  - num_bands = 16"
        echo "  - Projections: Bi:p (automatic)"
        echo "  - Disentanglement: ENABLED"
        echo ""
    fi
fi

# Path 2: 10 WFs, manual projections, no disentanglement
if [ "$PATH_TO_RUN" = "path2" ] || [ "$PATH_TO_RUN" = "both" ]; then
    print_header "Path 2: 10 WFs + Manual Projections"

    echo "Generating files with manual projections..."
    python3 tests/test_bismuth_full_workflow.py \
        --mode window \
        --window -5.0 3.0 \
        --seedname bismuth_path2_manual \
        --output-dir test_output

    if [ -f "test_output/bismuth_path2_manual.win" ]; then
        print_success "Base files generated"

        # Get atomic positions from .win
        ATOM1_POS=$(grep -A 2 "begin atoms_frac" test_output/bismuth_path2_manual.win | grep Bi | head -1 | awk '{print $2","$3","$4}')
        ATOM2_POS=$(grep -A 2 "begin atoms_frac" test_output/bismuth_path2_manual.win | grep Bi | tail -1 | awk '{print $2","$3","$4}')

        echo "Atom positions: $ATOM1_POS and $ATOM2_POS"

        # Replace projections with manual specification
        # Create temporary file with manual projections
        cat > /tmp/manual_proj.txt << EOF
! Manual projections: 5 orbitals × 2 spinor = 10 WFs
! Option A: 3 p-orbitals on atom 1, 2 on atom 2
f=$ATOM1_POS:px
f=$ATOM1_POS:py
f=$ATOM1_POS:pz
f=$ATOM2_POS:px
f=$ATOM2_POS:py

! Alternative options (uncomment to try):
! Option B: Symmetric p-orbitals
! f=$ATOM1_POS:px
! f=$ATOM1_POS:py
! f=$ATOM1_POS:pz
! f=$ATOM2_POS:px
! f=$ATOM2_POS:py
!
! Option C: Mixed s-p character
! f=$ATOM1_POS:s
! f=$ATOM1_POS:px
! f=$ATOM1_POS:py
! f=$ATOM2_POS:px
! f=$ATOM2_POS:py
EOF

        # Replace projections section
        awk '
            /begin projections/,/end projections/ {
                if (/begin projections/) {
                    print
                    system("cat /tmp/manual_proj.txt")
                } else if (/end projections/) {
                    print
                }
                next
            }
            {print}
        ' test_output/bismuth_path2_manual.win > test_output/bismuth_path2_manual.win.tmp
        mv test_output/bismuth_path2_manual.win.tmp test_output/bismuth_path2_manual.win

        # Increase iterations
        sed -i.bak 's/^num_iter = .*/num_iter = 5000/' test_output/bismuth_path2_manual.win

        rm /tmp/manual_proj.txt

        print_success "Path 2 configured"
        echo ""
        echo "Configuration:"
        echo "  - num_wann = 10"
        echo "  - num_bands = 10"
        echo "  - Projections: Manual (5 orbitals)"
        echo "  - Disentanglement: DISABLED"
        echo ""
    fi
fi

# Summary
print_header "Test Files Ready"
echo ""
echo "Generated configurations:"
if [ "$PATH_TO_RUN" = "path1" ] || [ "$PATH_TO_RUN" = "both" ]; then
    echo ""
    echo "Path 1: test_output/bismuth_path1_auto.*"
    echo "  - 12 WFs, 16 bands, Bi:p projections, disentanglement"
    ls -lh test_output/bismuth_path1_auto.{win,eig,amn,mmn,nnkp} 2>/dev/null | awk '{print "    "$9" ("$5")"}'
fi

if [ "$PATH_TO_RUN" = "path2" ] || [ "$PATH_TO_RUN" = "both" ]; then
    echo ""
    echo "Path 2: test_output/bismuth_path2_manual.*"
    echo "  - 10 WFs, 10 bands, manual projections, no disentanglement"
    ls -lh test_output/bismuth_path2_manual.{win,eig,amn,mmn,nnkp} 2>/dev/null | awk '{print "    "$9" ("$5")"}'
fi

echo ""
print_warning "Next Steps:"
echo "  1. Review .win files in test_output/"
echo "  2. Copy to cluster:"
echo "     scp test_output/bismuth_path*.{win,eig,amn,mmn,nnkp} cluster:~/localization_tests/"
echo "  3. Run on cluster:"
echo "     cd localization_tests"
echo "     wannier90.x bismuth_path1_auto"
echo "     wannier90.x bismuth_path2_manual"
echo "  4. Compare results:"
echo "     grep 'Final Spread' bismuth_path*.wout"
echo "     grep 'Omega' bismuth_path*.wout | grep 'SPRD'"
echo ""
print_success "Done!"
