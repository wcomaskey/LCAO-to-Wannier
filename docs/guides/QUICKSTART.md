# Quick Start: Testing with Bismuth_basis_40.out

## Step 1: Install (One-Time Setup)

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier
./quick_install.sh
```

This creates a virtual environment and installs numpy and scipy.

---

## Step 2: Run Tests

### Option A: Using the convenience script (EASIEST)

```bash
# Suggest optimal energy window
./run_tests.sh --mode suggest --target-wann 20

# Generate Wannier90 files with specific window
./run_tests.sh --mode window --window -5.0 3.0

# Use all bands (no selection)
./run_tests.sh --mode full --k-grid 4 4 1
```

The `run_tests.sh` script automatically activates the virtual environment for you!

---

### Option B: Manual activation

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python3 tests/test_bismuth_full_workflow.py --mode suggest --target-wann 20
python3 tests/test_bismuth_full_workflow.py --mode window --window -5.0 3.0

# Deactivate when done
deactivate
```

---

## What Each Mode Does

### `suggest` - Analyzes and suggests window
```bash
./run_tests.sh --mode suggest --target-wann 20
```
- Parses Bismuth file
- Auto-detects Fermi energy
- **Suggests optimal energy window** for target # of WFs
- Does NOT write files (analysis only)

**Output:**
```
✓ Window suggestion for 20 Wannier functions:
  Outer window: [-4.5, 2.5] eV
  Frozen window: [-4.0, 2.0] eV
```

---

### `window` - Generates Wannier90 files
```bash
./run_tests.sh --mode window --window -5.0 3.0 --k-grid 8 8 1
```
- Selects bands in energy window
- **Generates all 4 Wannier90 files:**
  - `test_output/bismuth_test.win`
  - `test_output/bismuth_test.eig`
  - `test_output/bismuth_test.amn`
  - `test_output/bismuth_test.mmn`

**Next step:**
```bash
cd test_output
wannier90.x bismuth_test
```

---

### `full` - Uses all bands
```bash
./run_tests.sh --mode full --k-grid 4 4 1
```
- Uses all 80 bands (good for testing)
- Generates Wannier90 files

---

## All Command-Line Options

```bash
./run_tests.sh [OPTIONS]

Options:
  --mode {suggest,window,full}  # What to do
  --k-grid NX NY NZ             # K-point grid (default: 4 4 1)
  --window E_MIN E_MAX          # Energy window in eV
  --target-wann N               # Target # of WFs
  --seedname NAME               # Output file prefix
  --output-dir DIR              # Where to save files
```

---

## Example Workflow

```bash
# 1. Install (first time only)
./quick_install.sh

# 2. See what window to use
./run_tests.sh --mode suggest --target-wann 20
# Output: Suggested window [-4.5, 2.5] eV

# 3. Generate files with suggested window
./run_tests.sh --mode window --window -4.5 2.5 --k-grid 8 8 1

# 4. Check output
ls -lh test_output/
head -50 test_output/bismuth_test.win

# 5. Run Wannier90 (if installed)
cd test_output
wannier90.x bismuth_test
```

---

## Troubleshooting

### Error: "No module named 'scipy'"
```bash
# Make sure you ran the install script
./quick_install.sh

# OR activate the venv manually
source venv/bin/activate
```

### Error: "venv not found"
```bash
# Run the install script
./quick_install.sh
```

### Error: "AttributeError"
```bash
# Update the test script (already fixed)
git pull  # or re-run quick_install.sh
```

---

## More Information

- **Full testing guide:** See `TESTING_GUIDE.md`
- **Test directory:** See `tests/README_TESTS.md`
- **Package README:** See `README.md`
- **Examples:** See `examples/` directory

---

## What's Happening Under the Hood

The test automatically:
1. ✅ Parses `tests/Bismuth_basis_40.out` (9.1 MB CRYSTAL output)
2. ✅ Extracts H(R), S(R) matrices, Fermi energy, atoms
3. ✅ Creates 80×80 spin-orbit coupled matrices
4. ✅ Solves eigenvalues at all k-points
5. ✅ Auto-detects Fermi level (from 46 electrons)
6. ✅ Analyzes energy bands
7. ✅ Selects bands based on window
8. ✅ Generates complete Wannier90 input (.win, .eig, .amn, .mmn)

**Result:** Ready-to-run Wannier90 input files!

---

## TL;DR (Too Long; Didn't Read)

```bash
# Install
./quick_install.sh

# Test
./run_tests.sh --mode suggest --target-wann 20
./run_tests.sh --mode window --window -5.0 3.0

# Run Wannier90
cd test_output && wannier90.x bismuth_test
```

**That's it!** 🚀
