# LCAO to Wannier90 Conversion Package

Version 1.0.2

## Overview

This package provides a complete computational workflow for converting Linear Combination of Atomic Orbitals (LCAO) calculation outputs into input files for Wannier90, enabling maximally localized Wannier function (MLWF) analysis.

## Features

- Parsing of real-space Hamiltonian H(R) and overlap S(R) matrices from LCAO outputs
- Fourier transformation to momentum space
- Solution of generalized eigenvalue problems at discrete k-points
- Generation of Wannier90-compatible file formats (.eig, .amn, .mmn)
- Numerical verification and validation routines
- Parallel processing support for large k-point grids

## Installation

### Standard Installation

```bash
cd lcao_wannier
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Quick Start

```python
from lcao_wannier import Wannier90Engine

# Initialize engine with your data
engine = Wannier90Engine(
    real_space_matrices=real_space_matrices,
    k_grid=(8, 8, 8),
    lattice_vectors=lattice_vectors,
    num_wann=20,
    seedname='material'
)

# Execute workflow
results = engine.run(parallel=True, verify=True)
```

## Package Structure

```
lcao_wannier/
├── lcao_wannier/          Main package
│   ├── __init__.py        Package exports
|   ├── band_selection.py  Band Selection Tools
│   ├── parser.py          LCAO output parsing
│   ├── kpoints.py         K-point grid generation
│   ├── fourier.py         Fourier transforms
│   ├── solver.py          Eigenvalue solver
│   ├── wannier90.py       File format writers
│   ├── verification.py    Numerical validation
│   ├── engine.py          Main coordination class
│   └── utils.py           Utility functions
├── tests/                 Test suite
│   ├── test_kpoints.py    K-point tests
│   ├── test_fourier.py    Fourier transform tests
│   ├── test_solver.py     Solver tests
│   ├── test_engine.py     Engine tests
│   └── test_integration.py Integration tests
├── examples/              Usage examples
│   ├── basic_usage.py     Simple example
│   └── advanced_workflow.py Complete workflow
├── setup.py               Installation configuration
├── pyproject.toml         Modern packaging metadata
└── requirements.txt       Dependencies
```

## Module Descriptions

### parser.py
Extracts overlap and Fock matrices from LCAO/CRYSTAL output files. Handles spin channels (alpha-alpha, beta-beta, alpha-beta, beta-alpha) and constructs spin-block matrices. Includes Hermiticity enforcement.

### kpoints.py
Generates Monkhorst-Pack k-point grids and neighbor lists with periodic boundary conditions. Provides index conversion utilities.

### fourier.py
Implements Fourier transforms between real and momentum space using phase factors exp(i 2π k·R). Includes phase factor precomputation for efficiency.

### solver.py
Solves the generalized eigenvalue problem H(k)C(k) = S(k)C(k)E(k) using scipy.linalg.eigh. Supports both sequential and parallel processing.

### wannier90.py
Writes Wannier90-compatible files (.eig, .amn, .mmn) following the official format specifications.

### verification.py
Provides numerical validation routines including Hermiticity checks and orthonormality verification.

### engine.py
Main coordination class (Wannier90Engine) that orchestrates the complete workflow from real-space matrices to Wannier90 files.

### utils.py
Helper functions for matrix preparation and data structure management.

## Testing

Run the complete test suite:

```bash
cd lcao_wannier
python tests/test_all.py
```

Run individual test modules:

```bash
python tests/test_kpoints.py
python tests/test_fourier.py
python tests/test_solver.py
python tests/test_engine.py
python tests/test_integration.py
```

All tests should pass. The test suite includes:
- K-point grid generation and neighbor lists
- Fourier transform accuracy and Hermiticity preservation
- Eigenvalue solver correctness
- Parallel vs. sequential consistency
- Complete workflow integration

## Usage Examples

### Basic Example

```python
import numpy as np
from lcao_wannier import Wannier90Engine

# Define lattice (cubic cell, 5 Angstrom)
lattice_vectors = np.eye(3) * 5.0

# Create simple tight-binding model
real_space_matrices = {}
real_space_matrices[(0,0,0)] = {
    'H': np.diag([0.0, 1.0, 2.0, 3.0]) + 0j,
    'S': np.eye(4) + 0j
}

# Add nearest-neighbor hopping
t = -0.5
for R in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
    real_space_matrices[R] = {
        'H': t * np.eye(4) + 0j,
        'S': 0.1 * np.eye(4) + 0j
    }

# Initialize and run
engine = Wannier90Engine(
    real_space_matrices=real_space_matrices,
    k_grid=(6, 6, 6),
    lattice_vectors=lattice_vectors,
    num_wann=4,
    seedname='material'
)

results = engine.run(parallel=True, verify=True)
```

### Integration with LCAO Parser

```python
from lcao_wannier import (
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine
)

# Parse LCAO output
with open('crystal.out', 'r') as f:
    lines = f.readlines()

matrices, lattice_vectors = parse_overlap_and_fock_matrices(lines)

# Create spin-block matrices
num_basis = next(iter(matrices['S_direct'].values())).shape[0]
H_full_list, S_full_list = create_spin_block_matrices(
    matrices['H_direct'],
    matrices['S_direct'],
    num_basis,
    lattice_vectors
)

# Prepare for engine
real_space_matrices = prepare_real_space_matrices(
    H_full_list,
    S_full_list,
    lattice_vectors
)

# Run engine
engine = Wannier90Engine(
    real_space_matrices=real_space_matrices,
    k_grid=(8, 8, 8),
    lattice_vectors=lattice_vectors,
    num_wann=20,
    seedname='material'
)

results = engine.run(parallel=True, verify=True)
```

## Output Files

The engine generates three files required by Wannier90:

### material.eig
Band energies at each k-point. Format:
```
band_index  k_index  energy_eV
```

### material.amn
Projection matrices A_mn(k) = <ψ_m(k)|g_n>. Format:
```
num_bands  num_kpoints  num_wann
band  wann  k_index  A_real  A_imag
```

### material.mmn
Overlap matrices M_mn(k,b) = <ψ_m(k)|ψ_n(k+b)>. Format:
```
num_bands  num_kpoints  num_neighbors
k_index  neighbor_k_index  b1 b2 b3
M_real  M_imag  (for all m,n pairs)
```

## Performance Considerations

- For k-grids with > 100 points, use parallel=True
- Memory usage scales as O(num_kpoints * num_orbitals^2)
- Parallel speedup is approximately linear up to num_kpoints
- Phase factor precomputation provides ~2x speedup for large systems

## Numerical Accuracy

The package maintains numerical precision through:
- Complex128 arithmetic throughout
- Hermiticity enforcement (error < 1e-15)
- Orthonormality verification (C† S C = I, error < 1e-14)
- Careful handling of phase factors in Fourier transforms

## API Reference

See `docs/API_REFERENCE.md` for complete API documentation.

## Citation

If you use this package in published research, please cite:

```
LCAO-Wannier: A computational engine for LCAO to Wannier90 conversion
Version 1.0.0 (2025)
```

## License

MIT License. See LICENSE file for details.

## Authors

William P. Comaskey 

## Support

For issues, questions, or contributions, see project documentation.
