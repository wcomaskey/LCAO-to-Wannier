# Tests Directory

This directory contains test files and test scripts for the LCAO-to-Wannier90 package.

## Test Data

**`Bismuth_basis_40.out`** (9.1 MB)
- 2D Bismuth system with spin-orbit coupling
- CRYSTAL23 output file
- 40 spatial basis functions → 80 spin-orbitals
- 15×15×1 k-point grid
- Contains H(R) and S(R) matrices for all R-vectors

## Test Scripts

### Unit Tests

| Script | Purpose | Run Command |
|--------|---------|-------------|
| `test_kpoints.py` | K-point grid generation | `python3 test_kpoints.py` |
| `test_fourier.py` | Fourier transforms | `python3 test_fourier.py` |
| `test_solver.py` | Eigenvalue solver | `python3 test_solver.py` |
| `test_engine.py` | Main engine class | `python3 test_engine.py` |
| `test_integration.py` | End-to-end workflow | `python3 test_integration.py` |
| `test_band_selection.py` | Band selection tools | `python3 test_band_selection.py` |
| `test_win_file.py` | .win file generation | `python3 test_win_file.py` |
| `test_all.py` | Run all unit tests | `python3 test_all.py` |

### Integration Tests

**`test_bismuth_full_workflow.py`** - Complete workflow test on real data

```bash
# Installation required first:
cd ..
./quick_install.sh
source venv/bin/activate

# Then run tests:
cd tests

# 1. Quick analysis (suggest window)
python3 test_bismuth_full_workflow.py --mode suggest --target-wann 20

# 2. Generate Wannier90 files
python3 test_bismuth_full_workflow.py --mode window --window -5.0 3.0

# 3. Full basis test
python3 test_bismuth_full_workflow.py --mode full --k-grid 4 4 1
```

## Quick Start

**First time setup:**
```bash
# From repository root
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Run quick install
./quick_install.sh

# Activate environment
source venv/bin/activate
```

**Run tests:**
```bash
cd tests

# Unit tests
python3 test_all.py

# Integration test
python3 test_bismuth_full_workflow.py --mode suggest
```

## Test Output

Tests create output in:
- `./test_output/` - Generated Wannier90 files
- Console output - Test results and analysis

Clean up:
```bash
rm -rf test_output/
```

## Expected Results

### Unit Tests
All tests should pass:
```
✓ test_kpoint_grid_generation
✓ test_fourier_transform_hermiticity
✓ test_eigenvalue_solver
...
Tests: 45 total, 45 passed, 0 failed
```

### Integration Test (Bismuth)
```
✓ Parsed calculation parameters:
  Fermi energy: -3.7xxx eV
  Electrons: 46
  K-grid from file: (15, 15, 1)

✓ Window suggestion for 20 Wannier functions:
  Outer window: [-4.5, 2.5] eV
  Frozen window: [-4.0, 2.0] eV
  Number of bands captured: 22
```

## Test Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| parser.py | ~90% | ✅ Comprehensive |
| kpoints.py | ~95% | ✅ Comprehensive |
| fourier.py | ~90% | ✅ Comprehensive |
| solver.py | ~85% | ✅ Good |
| engine.py | ~80% | ✅ Good |
| band_selection.py | ~85% | ✅ Good |
| wannier90.py | ~75% | ✅ Good |
| win_file.py | ~90% | ✅ Comprehensive |

## Troubleshooting

### Missing dependencies
```bash
pip install numpy scipy
```

### Import errors
```bash
# Make sure you're in venv
source ../venv/bin/activate

# Or install in development mode
cd ..
pip install -e .
```

### File not found
```bash
# Run from tests/ directory
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier/tests
python3 test_bismuth_full_workflow.py --mode suggest
```

## See Also

- `../TESTING_GUIDE.md` - Comprehensive testing guide
- `../README.md` - Package overview
- `../examples/` - Usage examples
