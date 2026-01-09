#!/usr/bin/env python3
"""
Script to improve Wannier function localization by updating .win file
with better settings and atomic projections.

Usage:
    python improve_localization.py test_output/bismuth_test.win

This will create test_output/bismuth_test_improved.win with:
1. Increased iterations (5000)
2. Atomic positions (if available)
3. Suggestions for projections based on system
"""

import sys
import argparse
from pathlib import Path


def improve_win_file(input_win: str, output_win: str = None):
    """
    Improve a .win file with better localization settings.

    Parameters
    ----------
    input_win : str
        Path to original .win file
    output_win : str, optional
        Path to output file. If None, creates *_improved.win
    """
    if output_win is None:
        input_path = Path(input_win)
        output_win = input_path.with_stem(f"{input_path.stem}_improved")

    # Read original file
    with open(input_win, 'r') as f:
        lines = f.readlines()

    # Track what we've seen
    has_atoms = False
    has_projections = False
    num_iter_line = None
    conv_window_line = None

    for i, line in enumerate(lines):
        if 'begin atoms' in line.lower():
            has_atoms = True
        if 'begin projections' in line.lower():
            has_projections = True
        if line.strip().startswith('num_iter'):
            num_iter_line = i
        if line.strip().startswith('conv_window'):
            conv_window_line = i

    # Prepare modifications
    modified_lines = lines.copy()

    # 1. Increase num_iter
    if num_iter_line is not None:
        modified_lines[num_iter_line] = "num_iter = 5000\n"
        print(f"✓ Updated num_iter to 5000 (was {lines[num_iter_line].strip()})")

    # 2. Increase conv_window
    if conv_window_line is not None:
        modified_lines[conv_window_line] = "conv_window = 5\n"
        print(f"✓ Updated conv_window to 5 (was {lines[conv_window_line].strip()})")

    # 3. Add suggestion comments
    improvements = []

    if not has_atoms:
        improvements.append(
            "\n! SUGGESTION: Add atomic positions for better analysis:\n"
            "! begin atoms_frac\n"
            "! Bi  0.000  0.000  0.510\n"
            "! Bi  0.000  0.000  0.490\n"
            "! end atoms_frac\n"
        )

    if not has_projections or 'random' in ''.join(lines).lower():
        improvements.append(
            "\n! CRITICAL: Replace 'random' projections with atomic orbitals!\n"
            "! This is likely why localization is poor.\n"
            "!\n"
            "! Option 1: Simple (Bismuth p-orbitals)\n"
            "! begin projections\n"
            "! Bi:p\n"
            "! end projections\n"
            "!\n"
            "! Option 2: Detailed (specify each orbital and atom)\n"
            "! begin projections\n"
            "! f=0.0,0.0,0.5:px,py,pz  ! Atom 1\n"
            "! f=0.0,0.0,-0.5:px,py,pz ! Atom 2\n"
            "! ! Add remaining projections to match num_wann\n"
            "! end projections\n"
            "!\n"
            "! Option 3: Hybrid orbitals\n"
            "! begin projections\n"
            "! Bi:sp3\n"
            "! end projections\n"
        )

    # Find where to insert improvements (after projections or after wannierization section)
    insert_position = None
    for i, line in enumerate(modified_lines):
        if 'end projections' in line.lower():
            insert_position = i + 1
            break

    if insert_position is None:
        # Insert after OUTPUT OPTIONS or WANNIERIZATION
        for i, line in enumerate(modified_lines):
            if 'OUTPUT OPTIONS' in line or 'WANNIERIZATION' in line:
                # Find next empty line
                for j in range(i, len(modified_lines)):
                    if modified_lines[j].strip() == '':
                        insert_position = j
                        break
                break

    if insert_position and improvements:
        modified_lines[insert_position:insert_position] = improvements
        print(f"✓ Added improvement suggestions to .win file")

    # Write modified file
    with open(output_win, 'w') as f:
        f.writelines(modified_lines)

    print(f"\n✓ Created improved .win file: {output_win}")
    print(f"\nNext steps:")
    print(f"1. Edit {output_win} and uncomment the projection lines")
    print(f"2. Choose appropriate atomic projections based on your system")
    print(f"3. Run: wannier90.x {Path(output_win).stem}")
    print(f"\nExpected improvement:")
    print(f"  - Omega_OD should decrease from ~560 to <100 Ang²")
    print(f"  - Individual spreads should be <20 Ang² for well-localized WFs")


def main():
    parser = argparse.ArgumentParser(
        description="Improve Wannier localization by updating .win file settings"
    )
    parser.add_argument(
        'input_win',
        help="Path to original .win file"
    )
    parser.add_argument(
        '-o', '--output',
        help="Path to output file (default: *_improved.win)"
    )

    args = parser.parse_args()

    if not Path(args.input_win).exists():
        print(f"Error: File not found: {args.input_win}")
        sys.exit(1)

    improve_win_file(args.input_win, args.output)


if __name__ == '__main__':
    main()
