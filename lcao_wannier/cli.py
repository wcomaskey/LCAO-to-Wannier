"""
CLI entry point for the lcao2wannier console script.

This is a thin shim that delegates to the main() function in
lcao_to_wannier90.py (installed as a top-level module via py-modules).

Usage via pip-installed entry point:
    lcao2wannier --stage 1 --input material.out --seedname material

Usage via direct script execution (backward-compatible):
    python lcao_to_wannier90.py --stage 1 --input material.out --seedname material
"""

from lcao_to_wannier90 import main


def cli_main():
    """Entry point for console_scripts."""
    main()


if __name__ == "__main__":
    cli_main()
