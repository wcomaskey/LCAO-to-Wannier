# Software Architecture

## System Overview

The LCAO-Wannier package implements a modular pipeline for converting LCAO calculation outputs to Wannier90 inputs. The architecture follows the separation of concerns principle, with each module handling a specific computational task.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface                            │
│                       (Wannier90Engine class)                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
┌───────────────────────┐ ┌──────────────┐ ┌──────────────────┐
│   Input Processing    │ │  Computation │ │ Output Generation│
│                       │ │              │ │                  │
│  ┌─────────────────┐  │ │ ┌──────────┐ │ │ ┌──────────────┐ │
│  │   parser.py     │  │ │ │kpoints.py│ │ │ │wannier90.py  │ │
│  │   utils.py      │  │ │ │fourier.py│ │ │ │              │ │
│  └─────────────────┘  │ │ │solver.py │ │ │ └──────────────┘ │
│                       │ │ └──────────┘ │ │                  │
└───────────────────────┘ └──────────────┘ └──────────────────┘
                                  │
                                  ▼
                      ┌────────────────────┐
                      │  verification.py   │
                      │  (Quality Control) │
                      └────────────────────┘
```

## Module Architecture

### Core Modules

#### 1. engine.py (Coordination Layer)

**Purpose**: Orchestrates the complete workflow

**Responsibilities**:
- Initialize system parameters
- Coordinate data flow between modules
- Manage parallel execution
- Provide high-level API

**Key Class**: `Wannier90Engine`

**Methods**:
```
__init__(real_space_matrices, k_grid, lattice_vectors, num_wann, seedname)
solve_all_kpoints(parallel, num_processes)
verify_results()
write_files()
run(parallel, verify)
```

**Dependencies**: All other modules

#### 2. parser.py (Input Layer)

**Purpose**: Parse LCAO/CRYSTAL output files

**Responsibilities**:
- Extract overlap and Fock matrices
- Handle spin channels
- Construct spin-block matrices
- Enforce Hermiticity

**Key Functions**:
```
parse_overlap_and_fock_matrices(lines)
create_spin_block_matrices(H_dict, S_dict, N_basis, lattice_vectors)
is_hermitian(matrix, tolerance)
make_hermitian(matrix)
```

**Dependencies**: NumPy

#### 3. kpoints.py (K-Space Discretization)

**Purpose**: Generate k-point grids and neighbor lists

**Responsibilities**:
- Monkhorst-Pack grid generation
- Periodic boundary conditions
- Neighbor list construction
- Index conversions

**Key Functions**:
```
generate_kpoint_grid(k_grid)
generate_neighbor_list(k_grid)
kpoint_index_to_grid(k_idx, k_grid)
grid_to_kpoint_index(i, j, k, k_grid)
```

**Dependencies**: NumPy

#### 4. fourier.py (Transform Layer)

**Purpose**: Fourier transforms between real and k-space

**Responsibilities**:
- Forward Fourier transform
- Phase factor computation
- Hermiticity preservation

**Key Functions**:
```
fourier_transform_to_kspace(k_point, real_space_matrices, lattice_vectors)
compute_phase_factors(kpoints, R_vectors)
```

**Dependencies**: NumPy

#### 5. solver.py (Eigenvalue Layer)

**Purpose**: Solve generalized eigenvalue problems

**Responsibilities**:
- Eigenvalue problem solution
- Parallel processing
- Result aggregation

**Key Functions**:
```
solve_generalized_eigenvalue_problem(H_k, S_k, num_wann)
solve_all_kpoints_sequential(...)
solve_all_kpoints_parallel(...)
```

**Dependencies**: NumPy, SciPy, multiprocessing

#### 6. wannier90.py (Output Layer)

**Purpose**: Write Wannier90 file formats

**Responsibilities**:
- .eig file generation
- .amn file generation
- .mmn file generation
- Format compliance

**Key Functions**:
```
write_eig_file(seedname, eigenvalues, num_kpoints, num_bands)
write_amn_file(seedname, projection_matrices, ...)
write_mmn_file(seedname, overlap_matrices, ...)
write_wannier90_files(seedname, results_dict)
```

**Dependencies**: NumPy

#### 7. verification.py (Quality Control)

**Purpose**: Numerical validation

**Responsibilities**:
- Hermiticity checks
- Orthonormality verification
- Error reporting

**Key Functions**:
```
verify_hermiticity(matrices, tolerance)
verify_orthonormality(eigenvectors, overlap_matrices, tolerance)
verify_eigenvalues(eigenvalues)
```

**Dependencies**: NumPy

#### 8. utils.py (Support Layer)

**Purpose**: Helper functions and utilities

**Responsibilities**:
- Data structure preparation
- Matrix operations
- Utility functions

**Key Functions**:
```
prepare_real_space_matrices(H_list, S_list, lattice_vectors)
organize_matrices_by_lattice_vector(matrices)
```

**Dependencies**: NumPy

## Data Flow

### 1. Input Phase

```
LCAO Output File
      │
      ▼
  parser.py ──┐
              │
              ▼
      Real-Space Matrices
      {R: {'H': H(R), 'S': S(R)}}
```

### 2. Computational Phase

```
Real-Space Matrices + K-Grid
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
    kpoints.py     fourier.py      solver.py
          │              │              │
          ▼              ▼              ▼
    K-Points         H(k), S(k)    E(k), C(k)
```

### 3. Output Phase

```
E(k), C(k), Neighbors
          │
          ▼
    wannier90.py
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
    .eig file      .amn file      .mmn file
```

## Design Patterns

### 1. Facade Pattern

The `Wannier90Engine` class acts as a facade, providing a simplified interface to the complex subsystem.

```python
class Wannier90Engine:
    def __init__(self, ...):
        # Initialize all subsystems
        self.kpoints = generate_kpoint_grid(k_grid)
        self.neighbors = generate_neighbor_list(k_grid)
        
    def run(self, parallel=True, verify=True):
        # Coordinate all operations
        self.solve_all_kpoints(parallel)
        if verify:
            self.verify_results()
        self.write_files()
```

### 2. Strategy Pattern

Parallel vs. sequential solving implemented as strategies:

```python
def solve_all_kpoints(self, parallel=True, num_processes=None):
    if parallel:
        return solve_all_kpoints_parallel(...)
    else:
        return solve_all_kpoints_sequential(...)
```

### 3. Builder Pattern

Matrix construction uses builder pattern:

```python
def prepare_real_space_matrices(H_list, S_list, lattice_vectors):
    matrices = {}
    for H, S, R in zip(H_list, S_list, R_vectors):
        matrices[R] = {'H': H, 'S': S}
    return matrices
```

## Error Handling

### Validation Strategy

1. **Input Validation** (parser.py)
   - Matrix dimensions
   - Hermiticity enforcement
   - Data type verification

2. **Computational Validation** (verification.py)
   - Hermiticity preservation
   - Orthonormality checks
   - Eigenvalue reality

3. **Output Validation** (wannier90.py)
   - File format compliance
   - Data completeness
   - Numerical range checks

### Exception Hierarchy

```
ValueError
  ├── MatrixDimensionError
  ├── HermitivityError
  └── OrthonormalityError

RuntimeError
  ├── ConvergenceError
  └── FileWriteError

IOError
  └── ParserError
```

## Performance Optimization

### 1. Vectorization

All matrix operations use NumPy's vectorized operations:
```python
H_k = np.sum([phase * H_R for phase, H_R in zip(phases, H_Rs)], axis=0)
```

### 2. Memory Management

- Pre-allocate arrays for k-point loops
- Use views instead of copies where possible
- Clear large intermediate arrays

### 3. Parallelization

- Embarrassingly parallel k-point loop
- Multiprocessing for CPU-bound tasks
- Minimal inter-process communication

### 4. Computational Complexity

| Operation | Complexity | Dominant for |
|-----------|-----------|--------------|
| Fourier transform | O(N_R N^2) | Small N |
| Eigenvalue solve | O(N^3) | Large N |
| Projection matrices | O(N^2 N_w) | Many Wannier functions |
| File I/O | O(N_k N_w^2) | Large k-grids |

## Testing Architecture

### Test Hierarchy

```
tests/
├── Unit Tests
│   ├── test_kpoints.py       (Grid generation)
│   ├── test_fourier.py        (Transforms)
│   └── test_solver.py         (Eigenvalues)
├── Integration Tests
│   └── test_integration.py    (Complete workflow)
└── Engine Tests
    └── test_engine.py         (High-level API)
```

### Test Strategy

1. **Unit Tests**: Each module tested independently with synthetic data
2. **Integration Tests**: Test data flow between modules
3. **Validation Tests**: Verify numerical accuracy
4. **Regression Tests**: Ensure consistent results across versions

## Configuration Management

### Runtime Configuration

```python
# K-point grid
k_grid = (n1, n2, n3)

# Number of Wannier functions
num_wann = N_w

# Parallel processing
parallel = True
num_processes = None  # Auto-detect

# Verification
verify = True
tolerance = 1e-12
```

### Build Configuration

```python
# setup.py
install_requires = [
    'numpy>=1.20.0',
    'scipy>=1.7.0',
]

python_requires = '>=3.7'
```

## Extensibility

### Adding New Parsers

Implement parser interface:
```python
def parse_new_format(lines):
    # Extract matrices
    return real_space_matrices, lattice_vectors
```

### Adding New Output Formats

Implement writer interface:
```python
def write_new_format(seedname, data):
    # Write format-specific files
    pass
```

### Adding New Verification Tests

Extend verification module:
```python
def verify_new_property(data, tolerance):
    # Implement verification
    return max_error
```

## Maintenance Considerations

### Code Quality

- PEP 8 compliance for all Python code
- Type hints for function signatures
- Comprehensive docstrings
- Unit test coverage > 90%

### Documentation

- Module-level docstrings
- Function-level docstrings with parameter descriptions
- Usage examples in docstrings
- Separate methodology documentation

### Version Control

- Semantic versioning (MAJOR.MINOR.PATCH)
- Changelog for all releases
- Tagged releases in version control
- Backward compatibility maintenance

## Deployment

### Installation Methods

1. **Development Mode**:
```bash
pip install -e .
```

2. **Standard Installation**:
```bash
pip install lcao_wannier
```

3. **From Source**:
```bash
python setup.py install
```

### Dependencies

- **Required**: numpy, scipy
- **Optional**: matplotlib (for visualization)
- **Development**: pytest, black, mypy

## Future Enhancements

### Planned Features

1. GPU acceleration for large systems
2. Adaptive k-point sampling
3. Band structure plotting
4. Symmetry analysis
5. Interface to other DFT codes

### Scalability Roadmap

1. **Current**: Single-node parallelization
2. **Phase 2**: Distributed computing (MPI)
3. **Phase 3**: GPU acceleration (CUDA/ROCm)
4. **Phase 4**: Cloud deployment

## Performance Benchmarks

### Reference System

- Intel Xeon 3.0 GHz (8 cores)
- 32 GB RAM
- Python 3.10, NumPy 1.24, SciPy 1.10

### Benchmark Results

| System Size | K-Grid | Time (seq) | Time (par) | Speedup |
|-------------|--------|------------|------------|---------|
| 50 orbitals | 4³     | 2.1 s      | 0.8 s      | 2.6x    |
| 50 orbitals | 8³     | 18.3 s     | 3.2 s      | 5.7x    |
| 100 orbitals| 4³     | 12.4 s     | 4.1 s      | 3.0x    |
| 100 orbitals| 8³     | 98.7 s     | 17.2 s     | 5.7x    |

Memory usage scales linearly with k-grid size.
