#!/usr/bin/env python3
"""
Validate collinear spin-polarized handling:
  * _detect_spin_type classifies restricted / collinear / soc from the header
  * create_nonsoc_full_matrices(spin_channel=...) selects ALPHA vs BETA
  * ALPHA selection == legacy default (first channel); BETA differs; S shared

Run: PYTHONPATH=<repo> python3 scripts/validate_spin_handling.py
"""
import os
import sys
import tempfile

import numpy as np

from lcao_wannier.parser import (
    parse_overlap_and_fock_matrices_streaming,
    create_nonsoc_full_matrices,
)
from lcao_wannier.utils import prepare_real_space_matrices

# Import the CLI module's helpers (importing does not run main()).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lcao_to_wannier90 as cli


def _block(header, diag):
    rows = [
        f"   1    {diag[0]: .7E}",
        f"   2     1.0000000E-01 {diag[1]: .7E}",
        f"   3     5.0000000E-02  1.0000000E-01 {diag[2]: .7E}",
        f"   4     2.0000000E-02  5.0000000E-02  1.0000000E-01 {diag[3]: .7E}",
    ]
    return (header + "\n\n"
            "                 1              2              3              4\n\n"
            + "\n".join(rows) + "\n\n")


def make_fixture(type_line, alpha_d, beta_d):
    txt = f" CRYSTAL - PROPERTIES - TYPE OF CALCULATION :  {type_line}\n"
    txt += " NUMBER OF AO                 4\n"
    txt += " DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)\n"
    txt += "         5.0000000    0.0000000    0.0000000\n"
    txt += "         0.0000000    5.0000000    0.0000000\n"
    txt += "         0.0000000    0.0000000    5.0000000\n\n"
    for R in ("  0  0  0", "  1  0  0", " -1  0  0"):
        txt += _block(f" OVERLAP MATRIX - CELL N.   1({R})", [1.0, 1.0, 1.0, 1.0])
    chans = [("ALPHA", alpha_d)]
    if beta_d is not None:
        chans.append(("BETA", beta_d))
    for spin, dd in chans:
        txt += f"    {spin}      ELECTRONS\n\n"
        for R in ("  0  0  0", "  1  0  0", " -1  0  0"):
            txt += _block(f" FOCK MATRIX - CELL N.   1({R})", dd)
    return txt


def write_tmp(txt):
    tf = tempfile.NamedTemporaryFile('w', suffix='.out', delete=False)
    tf.write(txt); tf.close()
    return tf.name


def rsm_for_channel(path, channel):
    H_R, S_R, lat, _ = parse_overlap_and_fock_matrices_streaming(path, promote_complex=False)
    H_full, S_full = create_nonsoc_full_matrices(H_R, S_R, lat, spin_channel=channel)
    return prepare_real_space_matrices(H_full, S_full, np.array(lat))


def main():
    print("Spin-handling validation")
    print("=" * 56)

    # 1. Detection
    coll = write_tmp(make_fixture("UNRESTRICTED OPEN SHELL", [-5, -3, -4, -3.5], [-4.8, -2.9, -3.9, -3.4]))
    restr = write_tmp(make_fixture("RESTRICTED CLOSED SHELL", [-5, -3, -4, -3.5], None))
    soc_txt = make_fixture("UNRESTRICTED OPEN SHELL", [-5, -3, -4, -3.5], [-4.8, -2.9, -3.9, -3.4])
    soc_txt = soc_txt.replace("DIRECT LATTICE", "DENSITY MATRIX FROM A TWO-COMPONENT SCF\n DIRECT LATTICE", 1)
    soc = write_tmp(soc_txt)
    try:
        assert cli._detect_spin_type(coll) == 'collinear', cli._detect_spin_type(coll)
        assert cli._detect_spin_type(restr) == 'restricted', cli._detect_spin_type(restr)
        assert cli._detect_spin_type(soc) == 'soc', cli._detect_spin_type(soc)
        print("  detection: collinear / restricted / soc  OK")

        # 2. Channel selection
        rsm_a = rsm_for_channel(coll, 'ALPHA_ALPHA')
        rsm_b = rsm_for_channel(coll, 'BETA_BETA')
        rsm_default = rsm_for_channel(coll, None)
        R0 = (0, 0, 0)
        assert not np.allclose(rsm_a[R0]['H'], rsm_b[R0]['H']), "ALPHA and BETA H identical!"
        assert np.allclose(rsm_a[R0]['H'], rsm_default[R0]['H']), "ALPHA != default(first)"
        assert np.allclose(rsm_a[R0]['S'], rsm_b[R0]['S']), "overlap should be shared"
        print("  channel select: ALPHA!=BETA, ALPHA==default, S shared  OK")
    finally:
        for p in (coll, restr, soc):
            os.unlink(p)

    print("=" * 56)
    print("PASS")


if __name__ == '__main__':
    main()
