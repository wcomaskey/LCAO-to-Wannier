# Changelog

All notable changes to the LCAO-to-Wannier90 package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-01-01

### Added - CRITICAL FEATURE
- **Complete Wannier90 .win file generation** - The package now generates ALL required Wannier90 input files
  - New `win_file.py` module for creating Wannier90 parameter files
  - `Wannier90WinConfig` dataclass for .win file configuration
  - `write_win_file()` function for generating .win files with proper formatting
  - `create_win_config_from_engine()` for automatic configuration from engine state
  - `parse_atoms_from_crystal_output()` for extracting atomic positions from CRYSTAL files

- **Enhanced engine.write_files() method**
  - Now writes .win file by default (write_win=True parameter)
  - Support for custom atomic positions, projections, and band structure paths
  - Automatic spinor detection and configuration
  - Integration with band selection for energy window parameters

- **Predefined high-symmetry k-point paths**
  - `KPATH_HEXAGONAL_2D` - For 2D hexagonal lattices (graphene, etc.)
  - `KPATH_SQUARE_2D` - For 2D square lattices
  - `KPATH_FCC` - For face-centered cubic crystals
  - `KPATH_BCC` - For body-centered cubic crystals
  - `KPATH_SIMPLE_CUBIC` - For simple cubic lattices

- **Comprehensive examples**
  - `examples/example_win_file.py` - Five detailed examples of .win file generation
  - Examples cover: basic usage, atoms & bands, manual config, CRYSTAL parsing, SOC systems

- **Complete test suite for .win files**
  - `tests/test_win_file.py` - 20+ tests covering all new functionality
  - Tests for configuration validation, file writing, engine integration, atom parsing

### Changed
- Updated `Wannier90Engine.write_files()` signature to accept .win parameters
- Engine now generates 4 files (.win, .eig, .amn, .mmn) instead of 3
- Updated completion messages to reflect .win file generation
- Version bumped from 1.1.1 to 1.2.0 across all configuration files

### Fixed
- **CRITICAL:** Package now provides complete Wannier90 workflow (previously missing .win file)
- Users can now run `wannier90.x` directly on generated files without manual .win creation

### Documentation
- Updated README.md to document .win file generation
- Added .win file to output files section
- Updated module descriptions to include win_file.py
- Enhanced feature list to highlight complete Wannier90 support

## [1.1.1] - 2024-12-XX

### Added
- Band selection and frozen window determination
- Automatic Fermi level detection
- Automatic projection orbital selection
- `band_selection.py` module with comprehensive band analysis tools

### Changed
- Enhanced `Wannier90Engine` with band window analysis capabilities
- Improved documentation with Band Selection Guide

## [1.0.2] - 2024-XX-XX

### Fixed
- Bug fixes in unit conversions
- File parsing improvements

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- Core LCAO to Wannier90 conversion functionality
- Parser for CRYSTAL/LCAO output files
- Fourier transform module
- Eigenvalue solver with parallel support
- Generation of .eig, .amn, .mmn files
- Comprehensive test suite
- Documentation (README, ARCHITECTURE, METHODOLOGY, PROJECT_STRUCTURE)

---

## Migration Guide: 1.1.1 → 1.2.0

### For existing users:

**No breaking changes** - existing code will continue to work. The .win file is now generated automatically by default.

### New capabilities:

```python
from lcao_wannier import Wannier90Engine, KPATH_HEXAGONAL_2D

engine = Wannier90Engine(...)
engine.solve_all_kpoints()

# NEW: Customize .win file generation
engine.write_files(
    write_win=True,  # Now default
    atoms=[('C', np.array([0.0, 0.0, 0.0]))],
    projections=['C:sp3'],
    spinors=True,
    bands_plot=True,
    kpoint_path=KPATH_HEXAGONAL_2D
)

# Result: material.win, material.eig, material.amn, material.mmn
# Can now run: wannier90.x material
```

### What's different:

1. **engine.write_files()** now has additional optional parameters
2. **engine.run()** now generates .win file automatically
3. Output message now includes .win file in the list
4. No changes required to existing scripts

---

## Planned for Version 1.3.0

- Validation tool for checking Wannier90 input completeness
- Enhanced error handling with custom exception hierarchy
- Checkpoint/restart functionality for large calculations
- Support for .nnkp file generation
- CLI interface: `lcao2wannier crystal.out --k-grid 8 8 8`

## Planned for Version 2.0.0

- Support for SIESTA output format
- Support for OpenMX output format
- Support for FHI-aims output format
- Symmetry exploitation for performance optimization
- GPU acceleration support with CuPy
- Visualization tools for band structures and Wannier functions
