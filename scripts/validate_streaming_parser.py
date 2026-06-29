#!/usr/bin/env python3
"""
Correctness gate for the streaming (low-memory) parser.

Asserts that ``parse_overlap_and_fock_matrices_streaming`` produces bit-identical
H_R_dict / S_R_dict to the legacy ``parse_overlap_and_fock_matrices`` followed by
the main script's organizing loop — on BOTH matrix formats:

  * REAL/IMAG complex format (examples/example_lcao_output.txt)
  * simple real format with "ALPHA/BETA ELECTRONS" (synthetic, like the Sc file)

Also checks the Design-C ``promote_complex=False`` path keeps values identical
while dropping to float64 when there is no imaginary part.

Run:
    PYTHONPATH=<repo> python3 scripts/validate_streaming_parser.py
"""
import os
import sys
import tempfile

import numpy as np

from lcao_wannier.parser import (
    parse_overlap_and_fock_matrices,
    parse_overlap_and_fock_matrices_streaming,
)


def legacy_dicts(filepath):
    """Replicate main script Step-1/Step-2: parse + organize into dicts."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    raw, lat = parse_overlap_and_fock_matrices(lines)
    H, S = {}, {}
    for mi in raw:
        R = tuple(mi['lattice_vector'])
        if mi['type'] == 'overlap':
            S[R] = mi['data']
        elif mi['type'] == 'fock':
            spin = mi.get('spin_channel', 0)
            H.setdefault(R, {})[spin] = mi['data']
    return H, S, lat


def assert_dicts_equal(H1, S1, H2, S2, label):
    assert set(S1) == set(S2), f"[{label}] S R-keys differ: {set(S1)^set(S2)}"
    for R in S1:
        assert np.array_equal(S1[R], S2[R]), f"[{label}] S[{R}] values differ"
    assert set(H1) == set(H2), f"[{label}] H R-keys differ: {set(H1)^set(H2)}"
    for R in H1:
        assert set(H1[R]) == set(H2[R]), f"[{label}] H[{R}] spin keys differ"
        for sp in H1[R]:
            assert np.array_equal(H1[R][sp], H2[R][sp]), \
                f"[{label}] H[{R}][{sp}] values differ"
    print(f"  [{label}] dicts identical: "
          f"{len(S2)} S-vectors, {len(H2)} H-vectors  OK")


SIMPLE_FIXTURE = """ NUMBER OF AO                 4
 SHRINK. FACT.(MONKH.)     3  3  1  SHRINKING FACTOR(GILAT NET)        6
 DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)
         5.0000000    0.0000000    0.0000000
         0.0000000    5.0000000    0.0000000
         0.0000000    0.0000000    5.0000000

 OVERLAP MATRIX - CELL N.   1(  0  0  0)

                 1              2              3              4

   1     1.0000000E+00
   2     1.0000000E-01  1.0000000E+00
   3     5.0000000E-02  1.0000000E-01  1.0000000E+00
   4     2.0000000E-02  5.0000000E-02  1.0000000E-01  1.0000000E+00

 OVERLAP MATRIX - CELL N.   2(  1  0  0)

                 1              2              3              4

   1     2.0000000E-01
   2     3.0000000E-02  2.0000000E-01
   3     1.0000000E-02  3.0000000E-02  2.0000000E-01
   4     5.0000000E-03  1.0000000E-02  3.0000000E-02  2.0000000E-01

    ALPHA      ELECTRONS

 FOCK MATRIX - CELL N.   1(  0  0  0)

                 1              2              3              4

   1    -5.0000000E+00
   2     5.0000000E-01 -3.0000000E+00
   3     2.0000000E-01  5.0000000E-01 -4.0000000E+00
   4     1.0000000E-01  2.0000000E-01  5.0000000E-01 -3.5000000E+00

 FOCK MATRIX - CELL N.   2(  1  0  0)

                 1              2              3              4

   1    -8.0000000E-01
   2     2.0000000E-01 -7.0000000E-01
   3     1.0000000E-01  2.0000000E-01 -6.0000000E-01
   4     5.0000000E-02  1.0000000E-01  2.0000000E-01 -5.0000000E-01

    BETA      ELECTRONS

 FOCK MATRIX - CELL N.   1(  0  0  0)

                 1              2              3              4

   1    -4.8000000E+00
   2     5.1000000E-01 -2.9000000E+00
   3     2.1000000E-01  5.1000000E-01 -3.9000000E+00
   4     1.1000000E-01  2.1000000E-01  5.1000000E-01 -3.4000000E+00

 FOCK MATRIX - CELL N.   2(  1  0  0)

                 1              2              3              4

   1    -7.9000000E-01
   2     2.1000000E-01 -6.9000000E-01
   3     1.1000000E-01  2.1000000E-01 -5.9000000E-01
   4     5.1000000E-02  1.1000000E-01  2.1000000E-01 -4.9000000E-01
"""


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(repo, 'examples', 'example_lcao_output.txt')

    print("Validating streaming parser vs legacy parser+organizer")
    print("=" * 60)

    n_ok = 0

    # 1. REAL/IMAG complex format
    if os.path.isfile(example):
        H1, S1, lat1 = legacy_dicts(example)
        H2, S2, lat2, hdr = parse_overlap_and_fock_matrices_streaming(example)
        assert_dicts_equal(H1, S1, H2, S2, "example REAL/IMAG")
        if lat1 is not None:
            assert np.allclose(np.array(lat1), np.array(lat2)), "lattice differs"
        n_ok += 1
    else:
        print(f"  (skipping example: {example} not found)")

    # 2. Simple real format (like Sc)
    with tempfile.NamedTemporaryFile('w', suffix='.out', delete=False) as tf:
        tf.write(SIMPLE_FIXTURE)
        simple_path = tf.name
    try:
        H1, S1, lat1 = legacy_dicts(simple_path)
        H2, S2, lat2, hdr = parse_overlap_and_fock_matrices_streaming(simple_path)
        assert_dicts_equal(H1, S1, H2, S2, "simple ALPHA/BETA")
        assert np.allclose(np.array(lat1), np.array(lat2)), "lattice differs"
        # header_lines should capture the pre-matrix region (NUMBER OF AO present)
        assert any('NUMBER OF AO' in l for l in hdr), "header_lines missing params"

        # 3. Design C: promote_complex=False keeps values, drops to float64
        H3, S3, _, _ = parse_overlap_and_fock_matrices_streaming(
            simple_path, promote_complex=False)
        for R in H1:
            for sp in H1[R]:
                assert np.allclose(H1[R][sp], H3[R][sp]), f"C: H[{R}][{sp}] value drift"
                assert H3[R][sp].dtype == np.float64, \
                    f"C: expected float64, got {H3[R][sp].dtype}"
        print("  [Design C promote_complex=False] values match, dtype float64  OK")
        n_ok += 1
    finally:
        os.unlink(simple_path)

    print("=" * 60)
    print(f"ALL CHECKS PASSED ({n_ok} format(s) validated)")


if __name__ == '__main__':
    main()
