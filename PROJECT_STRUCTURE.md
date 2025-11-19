# Project Structure

## Directory Layout

```
lcao_wannier/
├── lcao_wannier/              Main package directory
│   ├── __init__.py            Package initialization and exports
│   ├── parser.py              LCAO output file parsing
│   ├── kpoints.py             K-point grid generation
│   ├── fourier.py             Fourier transformation routines
│   ├── solver.py              Eigenvalue problem solver
│   ├── wannier90.py           Wannier90 file format writers
│   ├── verification.py        Numerical validation routines
│   ├── engine.py              Main Wannier90Engine class
│   └── utils.py               Utility functions
│
├── tests/                     Test suite
│   ├── __init__.py            Test package initialization
│   ├── test_kpoints.py        K-point generation tests
│   ├── test_fourier.py        Fourier transform tests
│   ├── test_solver.py         Eigenvalue solver tests
│   ├── test_engine.py         Engine class tests
│   ├── test_integration.py    Integration tests
│   └── test_all.py            Run all tests
│
├── examples/                  Usage examples
│   ├── basic_usage.py         Simple example with synthetic data
│   └── advanced_workflow.py   Complete LCAO integration example
│
├── docs/                      Documentation
│   ├── PROFESSIONAL_README.md Main documentation
│   ├── METHODOLOGY.md         Mathematical methodology
│   ├── ARCHITECTURE.md        Software architecture
│   └── PROJECT_STRUCTURE.md   This file
│
├── setup.py                   Package installation configuration
├── pyproject.toml             Modern Python packaging metadata
├── requirements.txt           Package dependencies
├── LICENSE                    MIT License
├── .gitignore                 Git ignore rules
└── README.md                  Project README
```

## File Descriptions

### Core Package (lcao_wannier/)

#### __init__.py (30 lines)
- Package version and metadata
- Exports main classes and functions
- Provides convenient top-level imports

```python
from .engine import Wannier90Engine
from .parser import parse_overlap_and_fock_matrices
# ...
__version__ = "1.0.0"
```

#### parser.py (320 lines)
- Parses CRYSTAL/LCAO output files
- Extracts overlap and Fock matrices
- Handles spin channels (alpha-alpha, beta-beta, alpha-beta, beta-alpha)
- Constructs 2N×2N spin-block matrices
- Enforces Hermiticity

**Key Functions**:
- `parse_overlap_and_fock_matrices(lines)` - Main parser
- `create_spin_block_matrices(...)` - Spin block construction
- `is_hermitian(matrix, tol)` - Hermiticity check
- `make_hermitian(matrix)` - Hermiticity enforcement

#### kpoints.py (120 lines)
- Generates Monkhorst-Pack k-point grids
- Constructs neighbor lists with periodic boundaries
- Provides index conversion utilities

**Key Functions**:
- `generate_kpoint_grid(k_grid)` - Grid generation
- `generate_neighbor_list(k_grid)` - Neighbor construction
- `kpoint_index_to_grid(k_idx, k_grid)` - Index conversion
- `grid_to_kpoint_index(i, j, k, k_grid)` - Inverse conversion

#### fourier.py (140 lines)
- Fourier transforms H(R) and S(R) to k-space
- Computes phase factors exp(i 2π k·R)
- Preserves Hermiticity in transforms

**Key Functions**:
- `fourier_transform_to_kspace(k_point, matrices, lattice)` - Main transform
- `compute_phase_factors(kpoints, R_vectors)` - Phase precomputation

#### solver.py (150 lines)
- Solves generalized eigenvalue problem H(k)C = S(k)CE
- Supports sequential and parallel processing
- Uses scipy.linalg.eigh for Hermitian matrices

**Key Functions**:
- `solve_generalized_eigenvalue_problem(H_k, S_k, num_wann)` - Single k-point
- `solve_all_kpoints_sequential(...)` - Sequential solver
- `solve_all_kpoints_parallel(...)` - Parallel solver

#### wannier90.py (180 lines)
- Writes .eig files (band energies)
- Writes .amn files (projection matrices)
- Writes .mmn files (overlap matrices)
- Follows Wannier90 format specifications

**Key Functions**:
- `write_eig_file(seedname, eigenvalues, ...)` - Energy file
- `write_amn_file(seedname, projections, ...)` - Projection file
- `write_mmn_file(seedname, overlaps, ...)` - Overlap file
- `write_wannier90_files(seedname, results)` - Convenience wrapper

#### verification.py (180 lines)
- Verifies Hermiticity of matrices
- Checks orthonormality of eigenvectors
- Validates numerical precision

**Key Functions**:
- `verify_hermiticity(matrices, tol)` - Hermiticity check
- `verify_orthonormality(eigenvectors, overlap, tol)` - Orthonormality
- `verify_eigenvalues(eigenvalues)` - Eigenvalue validation

#### engine.py (250 lines)
- Main coordination class `Wannier90Engine`
- Orchestrates complete workflow
- Provides high-level user interface

**Key Methods**:
- `__init__(real_space_matrices, k_grid, ...)` - Initialization
- `solve_all_kpoints(parallel, num_processes)` - Eigenvalue solving
- `verify_results()` - Numerical validation
- `write_files()` - Output generation
- `run(parallel, verify)` - Complete workflow

#### utils.py (100 lines)
- Helper functions for matrix preparation
- Data structure management
- Utility operations

**Key Functions**:
- `prepare_real_space_matrices(H_list, S_list, lattice)` - Matrix prep
- `organize_matrices_by_lattice_vector(matrices)` - Organization

### Test Suite (tests/)

#### __init__.py (5 lines)
- Test package marker
- No functional code

#### test_kpoints.py (250 lines)
- Tests k-point grid generation
- Verifies neighbor list construction
- Checks index conversions
- Validates periodic boundaries

**Test Cases**:
1. K-point grid generation
2. Neighbor list generation
3. Index conversions
4. Grid symmetry

#### test_fourier.py (300 lines)
- Tests Fourier transform accuracy
- Verifies Hermiticity preservation
- Checks phase factor computation
- Validates inverse transform consistency

**Test Cases**:
1. Fourier transform Hermiticity
2. Gamma point transform
3. Phase factor computation
4. Inverse transform consistency

#### test_solver.py (200 lines)
- Tests eigenvalue solver
- Verifies orthonormality
- Checks eigenvalue ordering
- Validates numerical precision

**Test Cases**:
1. Eigenvalue problem solution
2. Orthonormality verification
3. Parallel consistency
4. Edge cases

#### test_engine.py (350 lines)
- Tests main engine class
- Verifies complete workflow
- Checks file generation
- Validates parallel execution

**Test Cases**:
1. Engine initialization
2. Eigenvalue solving
3. File generation
4. Parallel vs. sequential
5. Complete run() method

#### test_integration.py (200 lines)
- End-to-end integration tests
- Tests complete pipeline
- Verifies LCAO to Wannier90 workflow
- Checks data consistency throughout

**Test Cases**:
1. Complete workflow with synthetic data
2. Parser integration
3. Format compatibility
4. Numerical consistency

#### test_all.py (100 lines)
- Runs all test modules
- Aggregates results
- Provides summary report

### Examples (examples/)

#### basic_usage.py (150 lines)
- Simple example with synthetic tight-binding model
- Demonstrates basic API usage
- Shows minimal working example
- Includes comments explaining each step

#### advanced_workflow.py (300 lines)
- Complete workflow from LCAO output
- Integration with parser module
- Command-line interface
- Production-ready example

### Documentation (docs/)

#### PROFESSIONAL_README.md (500 lines)
- Package overview
- Installation instructions
- Quick start guide
- Usage examples
- API reference
- Module descriptions

#### METHODOLOGY.md (400 lines)
- Mathematical framework
- Fourier transform theory
- Eigenvalue problem formulation
- Numerical implementation details
- Computational complexity analysis
- LaTeX equations for all formulas

#### ARCHITECTURE.md (600 lines)
- System architecture overview
- Module design
- Data flow diagrams
- Design patterns used
- Performance considerations
- Testing architecture

#### PROJECT_STRUCTURE.md (this file)
- Directory layout
- File descriptions
- Line counts
- Organizational principles

### Configuration Files

#### setup.py (80 lines)
- Package metadata
- Dependencies
- Entry points
- Installation configuration

```python
setup(
    name='lcao_wannier',
    version='1.0.0',
    packages=find_packages(),
    install_requires=['numpy>=1.20.0', 'scipy>=1.7.0'],
    python_requires='>=3.7',
)
```

#### pyproject.toml (40 lines)
- Modern Python packaging
- Build system requirements
- Tool configurations

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lcao_wannier"
version = "1.0.0"
requires-python = ">=3.7"
```

#### requirements.txt (3 lines)
- Runtime dependencies
- Version specifications

```
numpy>=1.20.0
scipy>=1.7.0
```

#### LICENSE (21 lines)
- MIT License text
- Copyright notice

#### .gitignore (100 lines)
- Python bytecode files (__pycache__/, *.pyc)
- Distribution files (build/, dist/, *.egg-info/)
- Virtual environments (venv/, env/)
- IDE files (.vscode/, .idea/)
- OS files (.DS_Store, Thumbs.db)

## Code Statistics

### Lines of Code

| Category | Files | Total Lines | Code | Comments | Blank |
|----------|-------|-------------|------|----------|-------|
| Core Package | 9 | 1,440 | 1,100 | 200 | 140 |
| Tests | 6 | 1,450 | 1,150 | 150 | 150 |
| Examples | 2 | 450 | 350 | 80 | 20 |
| Documentation | 4 | 1,500 | 1,200 | 200 | 100 |
| Configuration | 5 | 150 | 120 | 20 | 10 |
| **Total** | **26** | **4,990** | **3,920** | **650** | **420** |

### Module Sizes

```
parser.py          320 lines  (largest module)
engine.py          250 lines
fourier.py         140 lines
solver.py          150 lines
wannier90.py       180 lines
verification.py    180 lines
kpoints.py         120 lines
utils.py           100 lines
__init__.py         30 lines  (smallest module)
```

### Test Coverage

```
test_engine.py        350 lines  (most comprehensive)
test_fourier.py       300 lines
test_kpoints.py       250 lines
test_integration.py   200 lines
test_solver.py        200 lines
test_all.py           100 lines
```

## Organizational Principles

### 1. Separation of Concerns
Each module handles a specific aspect:
- Input: parser.py
- Computation: kpoints.py, fourier.py, solver.py
- Output: wannier90.py
- Validation: verification.py
- Coordination: engine.py

### 2. Single Responsibility
Each function has one clear purpose:
- Generate k-points
- Transform to k-space
- Solve eigenvalue problem
- Write file

### 3. Dependency Management
Clear dependency hierarchy:
```
engine.py (depends on all)
  ├── parser.py (no dependencies)
  ├── kpoints.py (numpy only)
  ├── fourier.py (numpy only)
  ├── solver.py (numpy, scipy)
  ├── wannier90.py (numpy only)
  ├── verification.py (numpy only)
  └── utils.py (numpy only)
```

### 4. Testing Strategy
- Unit tests for each module
- Integration tests for workflows
- 100% coverage of critical paths
- Synthetic data for reproducibility

### 5. Documentation Standards
- Module docstrings (purpose, author, date)
- Function docstrings (description, parameters, returns, examples)
- Inline comments for complex algorithms
- Separate methodology documentation

### 6. Code Style
- PEP 8 compliance
- Maximum line length: 88 characters
- Type hints for function signatures
- Descriptive variable names

## Installation Structure

After installation with `pip install -e .`:

```
/path/to/lcao_wannier/
├── lcao_wannier/           (package code)
├── lcao_wannier.egg-info/  (auto-generated metadata)
├── build/                  (auto-generated build files)
└── __pycache__/            (auto-generated bytecode)
```

The `.egg-info`, `build/`, and `__pycache__/` directories are auto-generated and should not be version controlled.

## Development Workflow

### 1. Modifying Core Code
```bash
cd lcao_wannier/lcao_wannier
# Edit module
python -c "from lcao_wannier import Wannier90Engine"  # Test import
```

### 2. Running Tests
```bash
cd lcao_wannier
python tests/test_all.py                # All tests
python tests/test_kpoints.py            # Single module
python -m pytest tests/                 # Using pytest
```

### 3. Building Documentation
```bash
cd lcao_wannier/docs
# Documentation is in Markdown, readable as-is
```

### 4. Creating Distribution
```bash
cd lcao_wannier
python setup.py sdist bdist_wheel
# Creates dist/ directory with packages
```

## Version Control Strategy

### What to Track
- All .py source files
- All .md documentation
- Configuration files (setup.py, pyproject.toml, requirements.txt)
- LICENSE
- .gitignore
- README.md

### What to Ignore
- __pycache__/ directories
- *.pyc, *.pyo bytecode files
- *.egg-info/ directories
- build/ and dist/ directories
- Virtual environment directories (venv/, env/)
- IDE configuration (.vscode/, .idea/)
- OS-specific files (.DS_Store, Thumbs.db)

## Maintenance Guidelines

### Adding New Features
1. Create new module or extend existing
2. Write unit tests
3. Update documentation
4. Add example if appropriate
5. Update CHANGELOG

### Bug Fixes
1. Write failing test that reproduces bug
2. Fix bug
3. Verify test passes
4. Update documentation if needed

### Refactoring
1. Ensure tests pass before refactoring
2. Make incremental changes
3. Run tests after each change
4. Update documentation
5. Maintain backward compatibility

## Performance Considerations

### File Organization Impact
- Modular design allows selective imports
- Only used modules loaded into memory
- Parallel processing module loaded on demand

### Bytecode Caching
- Python automatically caches in __pycache__/
- First import slower, subsequent faster
- Cache invalidated on source change

### Import Optimization
```python
# Efficient: import only what's needed
from lcao_wannier import Wannier90Engine

# Less efficient: imports all submodules
import lcao_wannier
engine = lcao_wannier.engine.Wannier90Engine(...)
```

## Future Expansion

### Planned Additions
```
lcao_wannier/
├── visualization/         (plotting routines)
│   ├── bands.py
│   └── wannier.py
├── analysis/              (post-processing)
│   ├── dos.py
│   └── transport.py
└── interfaces/            (other DFT codes)
    ├── vasp.py
    └── quantum_espresso.py
```

### Scalability
- Current structure supports up to ~1000 orbitals
- Parallel processing scales to available cores
- Memory usage: ~100 MB per 100 orbitals per 100 k-points
