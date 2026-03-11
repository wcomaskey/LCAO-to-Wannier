"""
WannSym Command-Line Interface

Standalone usage for symmetrizing wannier90_hr.dat files.

Usage:
    python -m wannsym wannier90_hr.dat [options]

Required input files (in working directory or specified via flags):
    - wannier90_hr.dat : Wannier90 real-space Hamiltonian output
    - poscar.in        : Crystal structure (VASP POSCAR-like format)
    - wann.in          : Wannier orbital definitions

Optional:
    - locaxis.in       : Local coordinate axes for atoms
"""

import argparse
import sys
import os

from .hamiltonian import HamiltonianData
from .symmetrize import symmetrize_hr, WannSymConfig
from .input_parsers import parse_poscar, parse_wann_in, parse_locaxis


def main():
    parser = argparse.ArgumentParser(
        prog='wannsym',
        description='Symmetrize wannier90_hr.dat using crystal symmetry (Reynolds operator)',
        epilog=(
            'Example:\n'
            '  python -m wannsym wannier90_hr.dat --poscar poscar.in --wann wann.in\n'
            '\n'
            'Original algorithm: Changming Yue (arxiv:1805.12148)\n'
            'Python 3 refactoring: William Comaskey'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'hr_file',
        help='Path to wannier90_hr.dat file'
    )
    parser.add_argument(
        '--poscar', default='poscar.in',
        help='Path to poscar.in (crystal structure). Default: poscar.in'
    )
    parser.add_argument(
        '--wann', default='wann.in',
        help='Path to wann.in (orbital definitions). Default: wann.in'
    )
    parser.add_argument(
        '--locaxis', default='locaxis.in',
        help='Path to locaxis.in (local axes). Default: locaxis.in'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output filename. Default: {hr_file}_nsymm{N}'
    )
    parser.add_argument(
        '--no-hermitize', action='store_true',
        help='Skip Hermitization step'
    )
    parser.add_argument(
        '--no-time-reversal', action='store_true',
        help='Skip time-reversal symmetry enforcement'
    )
    parser.add_argument(
        '--threshold', type=float, default=1e-9,
        help='Threshold for dropping small hoppings. Default: 1e-9'
    )
    parser.add_argument(
        '--tolerance', type=float, default=1e-5,
        help='Symmetry detection tolerance (spglib). Default: 1e-5'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.hr_file):
        print(f"Error: HR file not found: {args.hr_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.poscar):
        print(f"Error: POSCAR file not found: {args.poscar}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.wann):
        print(f"Error: wann.in file not found: {args.wann}", file=sys.stderr)
        sys.exit(1)

    # Read inputs
    if not args.quiet:
        print(f"WannSym: Post-Wannierization Hamiltonian Symmetrization")
        print(f"=" * 60)
        print(f"  HR file:  {args.hr_file}")
        print(f"  POSCAR:   {args.poscar}")
        print(f"  wann.in:  {args.wann}")
        print(f"  locaxis:  {args.locaxis}")
        print()

    # Parse crystal structure
    lattice, positions, atom_numbers, symbols = parse_poscar(args.poscar)
    if not args.quiet:
        print(f"  Crystal: {len(symbols)} atoms ({', '.join(set(symbols))})")

    # Parse Wannier orbital definitions
    orbital_types, is_spinor, atom_indices = parse_wann_in(args.wann)
    if not args.quiet:
        print(f"  Wannier atoms: {len(orbital_types)}, SOC: {is_spinor}")
        for i, (idx, orbs) in enumerate(zip(atom_indices, orbital_types)):
            print(f"    Atom {idx} ({symbols[idx-1]}): {orbs}")

    # Parse local axes (optional)
    local_axes = parse_locaxis(args.locaxis, natom_wan=len(orbital_types))

    # Read Hamiltonian
    if not args.quiet:
        print(f"\n  Loading {args.hr_file}...")
    hr_data = HamiltonianData.from_file(args.hr_file)
    if not args.quiet:
        print(f"  Loaded: {hr_data.norbs} orbitals, {hr_data.nrpt} R-points")

    # For WannSym standalone, the crystal structure in poscar.in may contain
    # more atoms than are in the Wannier projection. We need to use the full
    # crystal structure for symmetry detection but only the Wannier atoms
    # for the orbital structure.

    # Configure
    config = WannSymConfig(
        apply_hermitization=not args.no_hermitize,
        apply_time_reversal=not args.no_time_reversal and is_spinor,
        threshold=args.threshold,
        sym_tolerance=args.tolerance,
        verbose=not args.quiet,
    )

    # Run symmetrization
    if not args.quiet:
        print(f"\nSymmetrizing...")
        print(f"-" * 60)

    hr_sym, result = symmetrize_hr(
        hr_data,
        lattice_vectors=lattice,
        atom_positions_frac=positions,
        atom_numbers=atom_numbers,
        orbital_types_per_atom=orbital_types,
        has_soc=is_spinor,
        config=config,
    )

    # Write output
    output_file = args.output or f"{args.hr_file}_nsymm{result.nsymm}"
    nrpt_written = hr_sym.to_file(output_file, threshold=args.threshold)
    result.output_file = output_file

    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"  Output: {output_file}")
        print(f"  R-points: {result.nrpt_original} → {result.nrpt_symmetrized}")
        print(f"  Symmetry: {result.space_group} ({result.nsymm} operations)")
        print(f"  Max change: {result.max_change:.6e}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
