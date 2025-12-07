# LCAO to Wannier90: Band Selection Guide

This guide covers the `example_band_selection.py` script for converting LCAO calculations to Wannier90 format with automatic band selection.

## Overview

The script reads CRYSTAL output files, automatically extracts calculation parameters (Fermi energy, k-grid, electron count), and generates Wannier90 input files (`.eig`, `.amn`, `.mmn`) for a selected subset of bands.

## Quick Start

```bash
# Analyze band structure and get window suggestion
python examples/example_band_selection.py your_file.out --suggest-window

# Run with a specific energy window (eV, relative to E_F)
python examples/example_band_selection.py your_file.out --window -5.0 3.0

# Run with full basis (no band selection)
python examples/example_band_selection.py your_file.out
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `lcao_file` | Path to CRYSTAL output file (required) |
| `--k-grid NX NY NZ` | K-point grid (default: from file or 8 8 1) |
| `--window E_MIN E_MAX` | Energy window in eV relative to E_F |
| `--e-fermi EF` | Fermi energy in eV (default: from file) |
| `--num-electrons N` | Number of electrons (default: from file) |
| `--seedname NAME` | Output file prefix (default: bismuth) |
| `--suggest-window` | Analyze bands and suggest optimal window |
| `--target-wann N` | Target number of Wannier functions for suggestion |
| `--no-verify` | Skip verification checks |

## Workflow Modes

### 1. Suggest Window Mode

Analyzes the band structure and suggests an energy window:

```bash
python examples/example_band_selection.py calc.out --suggest-window --target-wann 30
```

Output includes:
- Band classification (frozen, partial, excluded)
- Suggested energy window for target number of bands
- Command to run with suggested parameters

### 2. Band Selection Mode

Runs with a specified energy window:

```bash
python examples/example_band_selection.py calc.out --window -8.0 2.0
```

The script will:
1. Parse the LCAO file
2. Identify bands fully within the window ("frozen" bands)
3. Select optimal projection orbitals
4. Generate Wannier90 files for selected bands only

### 3. Full Basis Mode

Uses all orbitals as Wannier functions (no selection):

```bash
python examples/example_band_selection.py calc.out
```

## Automatic Parameter Detection

The script parses these values from CRYSTAL output:

| Parameter | File Pattern |
|-----------|--------------|
| Fermi energy | `FERMI ENERGY -0.137E+00` (Hartree → eV) |
| Electrons | `N. OF ELECTRONS PER CELL 46` |
| K-grid | `SHRINK. FACT.(MONKH.) 15 15 1` |
| Basis size | `NUMBER OF AO 56` |

Command-line arguments override file values when specified.

## Understanding the Output

### Band Classification

```
Band classification:
  Bands 0-39:   E = [-160.50, -13.71] eV  → EXCLUDED (below window)
  Bands 40-41:  E = [-8.03, -5.84] eV    → PARTIAL (crosses boundary)
  Bands 42-51:  E = [-7.25, -0.56] eV    → FROZEN ✓
  Bands 52-111: E = [4.18, 1552.10] eV   → EXCLUDED (above window)
```

- **FROZEN**: Bands entirely within the window; included in output
- **PARTIAL**: Bands crossing window boundaries; excluded by default
- **EXCLUDED**: Bands outside the window

### Energy Units

- All energies in output are in **eV**
- CRYSTAL Fock matrices are in Hartree; conversion happens automatically
- Window specification (`--window`) is relative to E_F unless `--e-fermi 0` is used

## Example: 2D Bismuth with SOC

Typical workflow for a spin-orbit coupled 2D material:

```bash
# Step 1: Analyze band structure
python examples/example_band_selection.py Bi_soc.out --suggest-window

# Step 2: Choose window based on output (e.g., capture valence bands)
python examples/example_band_selection.py Bi_soc.out \
    --window -5.0 2.0 \
    --seedname bismuth_valence

# Step 3: Files are ready for Wannier90
ls bismuth_valence.*
# bismuth_valence.eig  bismuth_valence.amn  bismuth_valence.mmn
```

## Output Files

| File | Contents |
|------|----------|
| `seedname.eig` | Eigenvalues (eV) for each k-point and selected band |
| `seedname.amn` | Projection matrix A(k) = S(k)†C(k) |
| `seedname.mmn` | Overlap matrices M(k,b) between neighboring k-points |

## Notes

- For 2D systems, high-energy states (hundreds of eV) are vacuum states from the slab geometry; these are automatically excluded with appropriate window choice
- The number of Wannier functions equals the number of frozen bands
- Spin-orbit coupling doubles the basis size (2N × 2N matrices)

## Troubleshooting

**Window too narrow (0 frozen bands)**
- Expand the energy window
- Check that E_F is correctly set

**Partial bands at window edge**
- Adjust window boundaries to fully include or exclude bands
- The output suggests specific adjustments

**Memory issues with large systems**
- Reduce k-grid density
- Use `--no-verify` to skip verification steps