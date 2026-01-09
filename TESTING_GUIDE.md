# Testing Guide: Running Tests on Bismuth_basis_40.out

This guide shows you how to test the LCAO-to-Wannier90 package with the included Bismuth test file.

## Prerequisites

First, install the required dependencies:

```bash
# Option 1: Using virtual environment (RECOMMENDED)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

pip install numpy scipy

# Option 2: Install from package requirements
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier
pip install -e .

# Option 3: System-wide (if you have permission)
pip3 install --break-system-packages numpy scipy
```

## Quick Test Commands

Once dependencies are installed, run these tests:

### 1. **Window Suggestion Mode** (Recommended First Step)
Analyzes the band structure and suggests optimal energy windows:

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Make sure you're in the virtual environment!
source venv/bin/activate

# Then run the test
python3 tests/test_bismuth_full_workflow.py --mode suggest --target-wann 20

# OR use the convenience script (activates venv automatically)
./run_tests.sh --mode suggest --target-wann 20
```

**What this does:**
- Parses the Bismuth CRYSTAL output
- Solves eigenvalue problems on a test k-grid (4×4×1)
- Auto-detects Fermi energy
- Suggests energy window to capture ~20 Wannier functions
- **Does NOT write output files** (just analysis)

**Expected output:**
```
✓ Window suggestion for 20 Wannier functions:
  Outer window: [-X.XX, X.XX] eV
  Frozen window: [-X.XX, X.XX] eV
  Number of bands captured: ~20
```

---

### 2. **Energy Window Mode** (Band Selection)
Uses a specific energy window to select bands:

```bash
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -5.0 3.0 \
    --k-grid 4 4 1 \
    --seedname bismuth_valence
```

**What this does:**
- Selects bands within [-5.0, 3.0] eV window relative to E_F
- Generates all 4 Wannier90 files (.win, .eig, .amn, .mmn)
- Outputs to `test_output/bismuth_valence.*`

**Expected output:**
```
✓ WORKFLOW COMPLETED SUCCESSFULLY!
Output files in: ./test_output/
  bismuth_valence.win  - Wannier90 input parameters
  bismuth_valence.eig  - Band energies
  bismuth_valence.amn  - Projection matrices
  bismuth_valence.mmn  - Overlap matrices

Next steps:
  cd test_output
  wannier90.x bismuth_valence
```

---

### 3. **Full Basis Mode** (All Bands)
Uses all available bands without selection:

```bash
python3 tests/test_bismuth_full_workflow.py \
    --mode full \
    --k-grid 4 4 1 \
    --seedname bismuth_full
```

**What this does:**
- Uses all 80 bands (40 spatial × 2 for SOC)
- Good for testing, but may be too many WFs for practical use
- Generates complete Wannier90 input files

---

### 4. **Production Run** (Larger k-grid)
For actual research, use a denser k-grid:

```bash
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -4.0 2.0 \
    --k-grid 15 15 1 \
    --seedname bismuth_production \
    --output-dir ./wannier_output
```

**Notes:**
- Uses 15×15×1 grid (225 k-points) from original CRYSTAL calculation
- Enables automatic parallelization
- May take several minutes depending on your system

---

## Using the Existing example_band_selection.py

The package also includes a dedicated example script:

```bash
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier

# Suggest window
python3 examples/example_band_selection.py \
    tests/Bismuth_basis_40.out \
    --suggest-window \
    --target-wann 20

# Run with specific window
python3 examples/example_band_selection.py \
    tests/Bismuth_basis_40.out \
    --window -5.0 3.0 \
    --k-grid 8 8 1 \
    --seedname bismuth
```

---

## Understanding the Bismuth System

**File:** `tests/Bismuth_basis_40.out`
- **System:** 2D Bismuth with spin-orbit coupling (SOC)
- **Basis:** 40 spatial orbitals → 80 spin-orbitals (2N×2N matrices)
- **Original k-grid:** 15×15×1 (from CRYSTAL calculation)
- **Fermi energy:** Auto-detected from file (~-3.7 eV)
- **Electrons:** 46 (from CRYSTAL output)

**Expected band structure:**
- Low-lying core states (very negative energy)
- Valence bands near E_F
- Conduction bands above E_F
- High-energy vacuum states (2D slab geometry)

**Typical energy windows:**
- Valence only: `--window -5.0 -0.5`
- Valence + conduction: `--window -5.0 3.0`
- Wider window: `--window -8.0 5.0`

---

## Verifying Output Files

After generation, check the files:

```bash
cd test_output

# Check file sizes
ls -lh bismuth_*.{win,eig,amn,mmn}

# Inspect .win file
head -50 bismuth_valence.win

# Check number of k-points and bands
head -2 bismuth_valence.eig

# Verify .amn header
head -2 bismuth_valence.amn
```

**Expected sizes (for 4×4×1 grid, ~20 WFs):**
- `.win`: ~2 KB (text parameter file)
- `.eig`: ~5-10 KB (band energies)
- `.amn`: ~1-5 MB (projection matrices)
- `.mmn`: ~5-20 MB (overlap matrices)

---

## Running Wannier90

If you have Wannier90 installed:

```bash
cd test_output

# Run Wannier90
wannier90.x bismuth_valence

# Check output
ls -lh bismuth_valence*
# Should see: .wout, _hr.dat, _centres.xyz, etc.
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"

**Solution:**
```bash
# Create virtual environment
python3 -m venv ~/lcao_env
source ~/lcao_env/bin/activate
pip install numpy scipy

# Run tests
python3 tests/test_bismuth_full_workflow.py --mode suggest
```

### "No bands found in window"

**Solution:** Window is too narrow
```bash
# Try wider window
python3 tests/test_bismuth_full_workflow.py --mode suggest --target-wann 30
# Use suggested window
```

### Memory errors with large k-grids

**Solution:** Start with smaller grid
```bash
# Use 4×4×1 for testing
python3 tests/test_bismuth_full_workflow.py --k-grid 4 4 1 --mode window --window -5 3
```

### Output files not found

**Solution:** Check output directory
```bash
ls -la test_output/
# Files are in ./test_output/ by default
# Use --output-dir to change location
```

---

## Advanced Usage

### Custom Projections

Edit the test script to add custom projections:

```python
engine.write_files(
    atoms=atoms,
    projections=['Bi:sp3', 'Bi:p'],  # Custom projections
    spinors=True,
    bands_plot=True,
    kpoint_path=KPATH_HEXAGONAL_2D  # For 2D hexagonal Bi
)
```

### Band Structure Plotting

Enable band plotting in .win file:

```bash
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -5 3 \
    # Edit script to set bands_plot=True
```

### Parallel Execution

For very large k-grids:

```python
# Automatically enabled for grids with >16 points
# Controlled in test script:
use_parallel = np.prod(k_grid) > 16
engine.solve_all_kpoints(parallel=use_parallel)
```

---

## Complete Example Session

```bash
# 1. Setup environment
cd /Users/williamcomaskey/Documents/GitHub/LCAO-to-Wannier
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy

# 2. Analyze bands
python3 tests/test_bismuth_full_workflow.py --mode suggest --target-wann 20

# 3. Run with suggested window (example: -4.5 to 2.5 eV)
python3 tests/test_bismuth_full_workflow.py \
    --mode window \
    --window -4.5 2.5 \
    --k-grid 8 8 1 \
    --seedname bismuth_soc

# 4. Verify output
ls -lh test_output/bismuth_soc.*
head -100 test_output/bismuth_soc.win

# 5. Run Wannier90 (if installed)
cd test_output
wannier90.x bismuth_soc
```

---

## Summary of Test Modes

| Mode | Purpose | Outputs | Use Case |
|------|---------|---------|----------|
| `suggest` | Analyze bands | Console only | Determine optimal window |
| `window` | Band selection | 4 files (.win, .eig, .amn, .mmn) | Production runs |
| `full` | All bands | 4 files | Testing/debugging |

---

## Next Steps

1. ✅ Run `--mode suggest` to understand your band structure
2. ✅ Use suggested window with `--mode window`
3. ✅ Verify generated files
4. ✅ Run `wannier90.x` on the output
5. ✅ Analyze Wannier functions in `.wout` file

For more examples, see:
- `examples/example_band_selection.py` - Full featured example
- `examples/example_win_file.py` - .win file generation examples
- `Documentation/Band_Selection_Guide.md` - Detailed band selection guide
