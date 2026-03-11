#!/bin/bash
# Convenience script to run tests with proper environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: ./quick_install.sh"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if scipy is installed
python3 -c "import scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Dependencies not installed!"
    echo "Installing numpy and scipy..."
    pip install numpy scipy
fi

# Run the test with all arguments passed through
echo "Running test with environment activated..."
echo ""
python3 tests/test_bismuth_full_workflow.py "$@"
