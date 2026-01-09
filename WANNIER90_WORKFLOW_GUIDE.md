# Wannier90 Workflow Guide

## Complete Two-Stage Workflow for LCAO-to-Wannier90

This guide explains the **correct** workflow for generating Wannier functions from LCAO/CRYSTAL calculations.

---

## ⚠️ CRITICAL: The Order Matters!

You **must** follow this exact order:

1. **Stage 1**: Create `.win` parameter file
2. **Run Wannier90 preprocessing**: `wannier90.x -pp seedname`
3. **Stage 2**: Create `.eig`, `.amn`, `.mmn` data files using `.nnkp`
4. **Run Wannier90**: `wannier90.x seedname`

**Why this order?**
- Stage 1 creates the `.win` file that tells Wannier90 what you want
- Wannier90's `-pp` (preprocessing) analyzes your k-mesh and selects optimal neighbors, writing them to `.nnkp`
- Stage 2 reads `.nnkp` to use **exactly** the neighbor structure Wannier90 expects
- Without this order, the `.mmn` file will have the wrong neighbor structure and Wannier90 will fail

---

## Quick Start Example

```bash
# Step 1: Create .win file
python lcao_to_wannier90.py --stage 1 --input Bismuth_basis_40.out --seedname bismuth

# Step 2: Run Wannier90 preprocessing
wannier90.x -pp bismuth

# Step 3: Create data files using .nnkp neighbors
python lcao_to_wannier90.py --stage 2 --input Bismuth_basis_40.out --seedname bismuth

# Step 4: Run Wannier90
wannier90.x bismuth
```

---

## Detailed Workflow

### Prerequisites

1. Completed CRYSTAL/LCAO calculation with output file
2. Python environment with lcao_wannier package installed
3. Wannier90 installed and in your PATH

### Stage 1: Create Parameter File

**Purpose**: Generate the `.win` parameter file that defines your Wannier90 calculation.

**Command**:
```bash
python lcao_to_wannier90.py --stage 1 \
    --input <crystal_output> \
    --seedname <seedname> \
    [--window E_MIN E_MAX] \
    [--projections PROJ1 PROJ2 ...] \
    [--bands-plot]
```

**What it does**:
- Parses your LCAO/CRYSTAL output file
- Analyzes the band structure to determine how many Wannier functions to create
- Selects bands within the energy window
- Generates the `.win` parameter file with:
  - System parameters (lattice vectors, k-mesh)
  - Atomic positions
  - Energy window
  - Number of Wannier functions
  - Projections (initial guesses)
  - Optional band plotting settings

**Output**:
- `<seedname>.win` - Wannier90 parameter file

**Example**:
```bash
# Basic usage with default window [-5, 3] eV
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bismuth

# Custom energy window
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bismuth --window -6 2

# With specific projections
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bismuth \
    --projections "Bi:p" "Bi:s"

# Enable band structure plotting
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bismuth --bands-plot
```

**What happens**:
```
================================================================================
STAGE 1: Creating Wannier90 Parameter File (.win)
================================================================================
Input file: Bi.out
Seedname: bismuth

Step 1: Parsing CRYSTAL/LCAO output file...
✓ Parsed calculation parameters
Step 2: Creating spin-orbit coupled matrices...
✓ Created 112×112 SOC matrices
Step 3: Determining energy window...
✓ Using default window: [-5.00, 3.00] eV
Step 4: Initializing Wannier90 engine...
✓ Engine initialized
Step 5: Solving eigenvalue problems...
✓ Eigenvalue problems solved
Step 6: Analyzing band structure and selecting bands...
✓ Selected 10 bands for Wannier functions
Step 7: Extracting atomic positions...
✓ Found 2 atoms
Step 8: Writing bismuth.win file...
✓ Created: bismuth.win

================================================================================
STAGE 1 COMPLETE!
================================================================================
✓ Created: bismuth.win

NEXT STEP:
  Run Wannier90 preprocessing to generate neighbor information:
  → wannier90.x -pp bismuth
```

---

### Wannier90 Preprocessing

**Purpose**: Generate the `.nnkp` file containing the exact neighbor structure Wannier90 expects.

**Command**:
```bash
wannier90.x -pp <seedname>
```

**What it does**:
- Reads the `.win` parameter file
- Analyzes the k-point mesh
- Selects optimal neighbor shells based on k-space distances
- Writes the neighbor list to `.nnkp`

**Output**:
- `<seedname>.nnkp` - Nearest-neighbor k-points file

**Example**:
```bash
wannier90.x -pp bismuth
```

**What happens**:
```
 +---------------------------------------------------+
 |                   WANNIER90                       |
 +---------------------------------------------------+

 Running in serial mode

                     ------
                     SYSTEM
                     ------

 Lattice Vectors (Ang)
   a_1     3.718397  -2.146817   0.000000
   a_2     0.000000   4.293634   0.000000
   a_3     0.000000   0.000000 500.000000

                     ------------
                     K-POINT GRID
                     ------------

 Grid size = 15 x 15 x  1      Total points =  225

 The b-vectors are chosen automatically
 The following shells are used:   1,  6

 +----------------------------------------------------------------------------+
 | Completeness relation is fully satisfied [Eq. (B1), PRB 56, 12847 (1997)] |
 +----------------------------------------------------------------------------+

 Output file bismuth.nnkp written
```

**Key information**:
- Shows which shells were selected (e.g., "shells 1, 6")
- For this example: 8 neighbors per k-point (2 from shell 1, 6 from shell 6)
- Creates `bismuth.nnkp` with the exact neighbor list

---

### Stage 2: Create Data Files

**Purpose**: Generate the `.eig`, `.amn`, and `.mmn` data files using the neighbor structure from `.nnkp`.

**Command**:
```bash
python lcao_to_wannier90.py --stage 2 \
    --input <crystal_output> \
    --seedname <seedname> \
    [--window E_MIN E_MAX]
```

**What it does**:
- Parses the LCAO/CRYSTAL output (same as Stage 1)
- Reads the `.nnkp` file to get the exact neighbor list
- Solves eigenvalue problems at all k-points
- Generates data files using the `.nnkp` neighbors:
  - `.eig`: Band energies at each k-point
  - `.amn`: Projection matrices (overlaps between Bloch states and trial orbitals)
  - `.mmn`: Overlap matrices between neighboring k-points **using .nnkp neighbors**

**Output**:
- `<seedname>.eig` - Eigenvalues file
- `<seedname>.amn` - Projection matrices
- `<seedname>.mmn` - Overlap matrices

**Example**:
```bash
# Must use same window as Stage 1!
python lcao_to_wannier90.py --stage 2 --input Bi.out --seedname bismuth

# If you used custom window in Stage 1, use same window here
python lcao_to_wannier90.py --stage 2 --input Bi.out --seedname bismuth --window -6 2
```

**What happens**:
```
================================================================================
STAGE 2: Creating Wannier90 Data Files (.eig, .amn, .mmn)
================================================================================
Input file: Bi.out
Seedname: bismuth

✓ Found bismuth.nnkp
✓ Found bismuth.win

Step 1: Parsing CRYSTAL/LCAO output file...
✓ Parsed calculation parameters
Step 2: Creating spin-orbit coupled matrices...
✓ Created 112×112 SOC matrices
Step 3: Determining energy window...
✓ Using default window: [-5.00, 3.00] eV
Step 4: Initializing Wannier90 engine...
✓ Engine initialized
Step 5: Solving eigenvalue problems...
✓ Eigenvalue problems solved
Step 6: Analyzing band structure and selecting bands...
✓ Selected 10 bands for Wannier functions
Step 7: Writing data files using bismuth.nnkp neighbors...

✓ Reading neighbor list from bismuth.nnkp
  Using 8 neighbors per k-point from Wannier90

Writing Wannier90 files...
  ✓ bismuth.eig: 2250 eigenvalues
  ✓ bismuth.amn: 22500 matrix elements
  ✓ bismuth.mmn: 180000 matrix elements

================================================================================
STAGE 2 COMPLETE!
================================================================================
✓ Created: bismuth.eig
✓ Created: bismuth.amn
✓ Created: bismuth.mmn

All files generated with correct neighbor structure from .nnkp!

NEXT STEP:
  Run Wannier90 to generate maximally localized Wannier functions:
  → wannier90.x bismuth
```

**Critical**: Notice the line:
```
✓ Reading neighbor list from bismuth.nnkp
  Using 8 neighbors per k-point from Wannier90
```

This confirms that Stage 2 is using the **exact** neighbor structure that Wannier90 expects.

---

### Run Wannier90

**Purpose**: Generate the maximally localized Wannier functions.

**Command**:
```bash
wannier90.x <seedname>
```

**What it does**:
- Reads all input files (`.win`, `.eig`, `.amn`, `.mmn`, `.nnkp`)
- Verifies neighbor structure matches
- Performs iterative minimization to find MLWFs
- Outputs Wannier functions, band structure, etc.

**Output**:
- `<seedname>.wout` - Main output file
- `<seedname>_hr.dat` - Hamiltonian in Wannier basis
- `<seedname>_centres.xyz` - Wannier function centers
- `<seedname>_band.dat` - Interpolated band structure (if enabled)
- And more...

**Example**:
```bash
wannier90.x bismuth
```

**Success indicators**:
```
 Reading overlaps from bismuth.mmn
 ✓ Overlap matrices read successfully

 Initial State
  WF centre and spread    1  ( ...

 Final State
  WF centre and spread    1  ( ...
  Sum of centres and spreads ( ...

 Wannier Function Plots written
```

If you see an error like:
```
 bismuth.mmn has not the right number of nearest neighbours
```

This means you **skipped Step 2** (running `wannier90.x -pp`) or didn't use `--stage 2` to regenerate files with `.nnkp` neighbors!

---

## File Summary

After completing all 4 steps, you should have:

### Created by Stage 1:
- `<seedname>.win` - Wannier90 input parameters

### Created by Wannier90 preprocessing:
- `<seedname>.nnkp` - Nearest-neighbor k-points (neighbor list)

### Created by Stage 2:
- `<seedname>.eig` - Band energies
- `<seedname>.amn` - Projection matrices
- `<seedname>.mmn` - Overlap matrices (using `.nnkp` neighbors!)

### Created by Wannier90:
- `<seedname>.wout` - Output file with results
- `<seedname>_hr.dat` - Hamiltonian in Wannier basis
- `<seedname>_centres.xyz` - Wannier centers
- `<seedname>_band.dat` - Band structure (if plotting enabled)
- And more...

---

## Common Options

### Energy Window

The energy window determines which bands are included in the Wannier function construction:

```bash
# Default: [-5, 3] eV relative to Fermi level
--window -5 3

# Narrow window for specific bands
--window -2 1

# Wide window for more bands
--window -8 5
```

**Important**: Use the **same window** in both Stage 1 and Stage 2!

### Projections

Initial guesses for the Wannier functions:

```bash
# Automatic (random)
# No --projections flag

# Specific orbital projections
--projections "Bi:sp3"

# Multiple projections
--projections "Bi:p" "Bi:s"
```

See the Wannier90 manual for projection syntax.

### Band Plotting

Enable Wannier90 band structure interpolation:

```bash
--bands-plot
```

This adds the necessary settings to `.win` for band plotting. You'll need to define a k-path in the `.win` file or use defaults.

### Parallel vs Serial

```bash
# Use parallel computation (default, recommended)
python lcao_to_wannier90.py --stage 1 --input file.out --seedname name

# Force serial computation (slower, for debugging)
python lcao_to_wannier90.py --stage 1 --input file.out --seedname name --no-parallel
```

---

## Troubleshooting

### Error: "bismuth.nnkp not found"

**Problem**: You ran Stage 2 before running `wannier90.x -pp`.

**Solution**: Follow the correct order:
1. Stage 1
2. `wannier90.x -pp seedname`  ← Don't skip this!
3. Stage 2
4. `wannier90.x seedname`

---

### Error: "bismuth.mmn has not the right number of nearest neighbours"

**Problem**: The `.mmn` file was created without using the `.nnkp` neighbor list.

**Possible causes**:
1. You ran the old workflow without Stage 1 and Stage 2
2. You manually created `.mmn` without reading `.nnkp`
3. You used an old version of the code

**Solution**: Follow the two-stage workflow:
```bash
# Delete old files
rm bismuth.eig bismuth.amn bismuth.mmn

# Redo Stage 2 to regenerate with correct neighbors
python lcao_to_wannier90.py --stage 2 --input Bi.out --seedname bismuth
```

---

### Error: "No bands found in the energy window"

**Problem**: Your energy window doesn't contain any complete bands.

**Solution**: Adjust the window:
```bash
# Try a wider window
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bismuth --window -8 5
```

Check the band structure to see which energies contain the bands you want.

---

### Different energy windows in Stage 1 and Stage 2

**Problem**: You used different `--window` values in the two stages.

**Effect**: The band selection will be different, causing inconsistencies.

**Solution**: Always use the **same** `--window` in both stages!

```bash
# CORRECT:
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bi --window -6 2
wannier90.x -pp bi
python lcao_to_wannier90.py --stage 2 --input Bi.out --seedname bi --window -6 2
wannier90.x bi

# WRONG:
python lcao_to_wannier90.py --stage 1 --input Bi.out --seedname bi --window -6 2
wannier90.x -pp bi
python lcao_to_wannier90.py --stage 2 --input Bi.out --seedname bi --window -5 3  # Different!
```

---

## Advanced Usage

### Checking the neighbor list

To see what neighbors Wannier90 selected:

```bash
grep -A 10 "begin nnkpts" bismuth.nnkp
```

Output:
```
begin nnkpts
   8
     1     1      0   0   1
     1     1      0   0  -1
     1     2      0   0   0
     1    16      0   0   0
     ...
```

The first line shows the number of neighbors per k-point (8 in this example).

### Verifying .mmn file

Check that your `.mmn` file has the correct number of neighbors:

```bash
head -2 bismuth.mmn
```

Output:
```
Created by LCAO-to-Wannier90 Engine
   10   225     8
```

The third number (8) must match the number in the `.nnkp` file!

---

## Complete Example: Bismuth

Here's a complete example from start to finish:

```bash
# Starting with Bismuth_basis_40.out from CRYSTAL

# Stage 1: Create .win file
python lcao_to_wannier90.py \
    --stage 1 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -5 3

# Output: bismuth.win

# Wannier90 preprocessing
wannier90.x -pp bismuth

# Output: bismuth.nnkp (8 neighbors per k-point for this system)

# Stage 2: Create data files
python lcao_to_wannier90.py \
    --stage 2 \
    --input tests/Bismuth_basis_40.out \
    --seedname bismuth \
    --window -5 3

# Output: bismuth.eig, bismuth.amn, bismuth.mmn

# Run Wannier90
wannier90.x bismuth

# Output: bismuth.wout, bismuth_hr.dat, bismuth_centres.xyz, etc.

# Check results
cat bismuth.wout
```

Expected output from `bismuth.wout`:
```
 Final State
  WF centre and spread    1  ( ... )
  WF centre and spread    2  ( ... )
  ...
  WF centre and spread   10  ( ... )
  Sum of centres and spreads ( ... )

 Wannier centres written to file bismuth_centres.xyz
```

Success! You now have maximally localized Wannier functions for Bismuth.

---

## Script Help

For quick reference:

```bash
python lcao_to_wannier90.py --help
```

Shows all available options and examples.

---

## Summary: The Four-Step Workflow

Remember this sequence:

```
1. Create .win     →  python lcao_to_wannier90.py --stage 1 -i file.out -s name
                    ↓
2. Preprocess      →  wannier90.x -pp name
                    ↓
3. Create data     →  python lcao_to_wannier90.py --stage 2 -i file.out -s name
                    ↓
4. Run Wannier90   →  wannier90.x name
```

Following this order ensures that:
- ✓ The `.nnkp` file is generated with optimal neighbors
- ✓ The `.mmn` file uses exactly those neighbors
- ✓ Wannier90 runs successfully without neighbor errors
- ✓ You get correct, maximally localized Wannier functions

---

## Questions?

If you encounter issues:
1. Check that you followed the 4-step order exactly
2. Verify the `.nnkp` file exists before Stage 2
3. Check that the neighbor count in `.mmn` matches `.nnkp`
4. Ensure you used the same `--window` in both stages

For the technical details of why this workflow is necessary, see `TECHNICAL_NOTES.md`.
