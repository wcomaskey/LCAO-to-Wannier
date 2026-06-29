#!/usr/bin/env python3
"""Development harness for the EXACT analytic GTO MMN method.

Conventions (CRYSTAL23 manual, Ch 17/18 + Basis-set sec 2.2):
  * Bloch phase e^{ik.g}, g = CELL vector (Convention II); AO centred at nucleus.
  * AOs = contractions of individually-normalized GTFs; the .out's printed
    coefficients work with UNNORMALIZED primitives e^{-a r^2}, contracted AO then
    normalized to unit self-overlap (validated against printed S(0)).
  * Order of internal storage:
      S: s
      P: x, y, z
      D (real solid harmonics): 2z^2-x^2-y^2, xz, yz, x^2-y^2, xy

Milestone 2: reconstruct the full S(R=0) from the parsed basis and match the
printed 'OVERLAP MATRIX - CELL N. 1( 0 0 0)'.

Run:  python3 scripts/analytic_mmn_dev.py calculations/BN_1C.out
"""
import sys
import re
import numpy as np
from math import pi, sqrt

# d real-solid-harmonic -> Cartesian (powers): coefficient list
DCART = [
    [((0, 0, 2), 2.0), ((2, 0, 0), -1.0), ((0, 2, 0), -1.0)],  # 2z^2-x^2-y^2
    [((1, 0, 1), 1.0)],                                         # xz
    [((0, 1, 1), 1.0)],                                         # yz
    [((2, 0, 0), 1.0), ((0, 2, 0), -1.0)],                      # x^2-y^2
    [((1, 1, 0), 1.0)],                                         # xy
]
PCART = [[((1, 0, 0), 1.0)], [((0, 1, 0), 1.0)], [((0, 0, 1), 1.0)]]
SCART = [[((0, 0, 0), 1.0)]]
LCART = {0: SCART, 1: PCART, 2: DCART}


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def parse_basis(lines):
    """Return (atoms, templates). atoms = [(idx, sym, center_bohr)] for ALL atoms;
    templates[sym] = [shell dicts] (l, prims) from the first atom carrying that
    sym's shells. prims = [(exp, coef_for_this_l), ...]."""
    LMAP = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}
    COL = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3}
    start = next(i for i, ln in enumerate(lines)
                 if 'LOCAL ATOMIC FUNCTIONS BASIS SET' in ln)
    atoms = []
    sym_shells = {}      # sym -> list of shells (built from first such atom)
    cur_sym = None
    cur_center = None
    cur_shell = None
    atom_re = re.compile(r'^\s*(\d+)\s+([A-Z][a-z]?)\s+'
                         r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
    shell_re = re.compile(r'^\s*(\d+)\s*-?\s*(\d*)\s+([SPDFG])\s*$')
    prim_re = re.compile(r'^\s*(-?\d\.\d+E[+-]\d+)\s+(-?\d\.\d+E[+-]\d+)\s+'
                         r'(-?\d\.\d+E[+-]\d+)\s+(-?\d\.\d+E[+-]\d+)\s*$')
    for ln in lines[start + 2:]:
        if 'OVERLAP MATRIX' in ln:
            break
        m = atom_re.match(ln)
        if m:
            cur_sym = m.group(2)
            cur_center = np.array([float(m.group(3)), float(m.group(4)),
                                   float(m.group(5))])
            atoms.append((int(m.group(1)), cur_sym, cur_center))
            cur_shell = None
            continue
        m = shell_re.match(ln)
        if m and cur_sym is not None:
            l = LMAP[m.group(3)]
            cur_shell = dict(l=l, prims=[])
            sym_shells.setdefault(cur_sym, [])
            # only record template for the FIRST atom of this sym
            if atoms[-1][0] == _first_atom_of_sym(atoms, cur_sym):
                sym_shells[cur_sym].append(cur_shell)
            continue
        m = prim_re.match(ln)
        if m and cur_shell is not None:
            vals = [float(m.group(i)) for i in range(1, 5)]
            cur_shell['prims'].append((vals[0], vals[COL[cur_shell['l']]]))
    return atoms, sym_shells


def _first_atom_of_sym(atoms, sym):
    for idx, s, _ in atoms:
        if s == sym:
            return idx
    return None


def build_aos(atoms, templates):
    """Instantiate the per-symbol shell templates at every atom -> ordered AO
    list. Each AO: dict(center, terms=[((lx,ly,lz),ang_coef)], prims, norm)."""
    aos = []
    for idx, sym, center in atoms:
        for sh in templates[sym]:
            for comp in LCART[sh['l']]:
                aos.append(dict(center=center, terms=comp, prims=sh['prims']))
    # normalize each AO to unit self-overlap
    for ao in aos:
        ao['norm'] = 1.0
        ao['norm'] = 1.0 / sqrt(ao_overlap(ao, ao, np.zeros(3)))
    return aos


# ----------------------------------------------------------------------------
# Cartesian Gaussian overlap (b=0)
# ----------------------------------------------------------------------------
def _overlap_1d(a, b, PA, PB, la, lb):
    p = a + b
    S = np.zeros((la + 2, lb + 2))
    S[0, 0] = 1.0
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                S[i, j] = PA * S[i - 1, j] + (1.0 / (2 * p)) * (
                    (i - 1) * S[i - 2, j] + j * S[i - 1, j - 1])
            else:
                S[i, j] = PB * S[i, j - 1] + (1.0 / (2 * p)) * (
                    i * S[i - 1, j - 1] + (j - 1) * S[i, j - 2])
    return S[la, lb]


def _prim_cart(a, A, lA, b, B, lB):
    p = a + b
    P = (a * A + b * B) / p
    pre = np.exp(-a * b / p * np.dot(A - B, A - B)) * (pi / p) ** 1.5
    return pre * (_overlap_1d(a, b, P[0] - A[0], P[0] - B[0], lA[0], lB[0])
                  * _overlap_1d(a, b, P[1] - A[1], P[1] - B[1], lA[1], lB[1])
                  * _overlap_1d(a, b, P[2] - A[2], P[2] - B[2], lA[2], lB[2]))


def _overlap_1d_cplx(p, PA, PB, la, lb):
    """Same Hermite recursion as _overlap_1d but PA/PB may be complex."""
    S = np.zeros((la + 2, lb + 2), dtype=complex)
    S[0, 0] = 1.0
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                S[i, j] = PA * S[i - 1, j] + (1.0 / (2 * p)) * (
                    (i - 1) * S[i - 2, j] + j * S[i - 1, j - 1])
            else:
                S[i, j] = PB * S[i, j - 1] + (1.0 / (2 * p)) * (
                    i * S[i - 1, j - 1] + (j - 1) * S[i, j - 2])
    return S[la, lb]


def _prim_cart_mom(a, A, lA, b, B, lB, bvec):
    """<g_a(r-A)| e^{-i bvec.r} | g_b(r-B)>, analytic. Gaussian product at P with
    exponent p; e^{-i b.r} shifts the centre to the COMPLEX point P' = P - i b/2p
    and adds scalar factors e^{-i b.P} e^{-|b|^2/4p}. Reduces to _prim_cart at b=0."""
    p = a + b
    P = (a * A + b * B) / p
    Pp = P - 1j * bvec / (2 * p)
    pre = (np.exp(-a * b / p * np.dot(A - B, A - B)) * (pi / p) ** 1.5
           * np.exp(-1j * np.dot(bvec, P)) * np.exp(-np.dot(bvec, bvec) / (4 * p)))
    return pre * (_overlap_1d_cplx(p, Pp[0] - A[0], Pp[0] - B[0], lA[0], lB[0])
                  * _overlap_1d_cplx(p, Pp[1] - A[1], Pp[1] - B[1], lA[1], lB[1])
                  * _overlap_1d_cplx(p, Pp[2] - A[2], Pp[2] - B[2], lA[2], lB[2]))


def ao_overlap(ao_i, ao_j, Rj):
    """Overlap <ao_i (cell 0) | ao_j (shifted by Rj, Bohr)>. Uses stored norms
    if present (else raw)."""
    Ci = ao_i['center']
    Cj = ao_j['center'] + Rj
    tot = 0.0
    for (li, ai) in ao_i['terms']:
        for (lj, aj) in ao_j['terms']:
            s = 0.0
            for (ea, ca) in ao_i['prims']:
                for (eb, cb) in ao_j['prims']:
                    s += ca * cb * _prim_cart(ea, Ci, li, eb, Cj, lj)
            tot += ai * aj * s
    return tot * ao_i.get('norm', 1.0) * ao_j.get('norm', 1.0)


def main():
    path = sys.argv[1]
    with open(path, errors='ignore') as f:
        lines = f.readlines()
    atoms, templates = parse_basis(lines)
    print(f"Atoms: {[(a[0], a[1]) for a in atoms]}")
    for sym, shs in templates.items():
        print(f"  template {sym}: {[s['l'] for s in shs]} "
              f"({sum(2*s['l']+1 if s['l']<2 else 5 for s in shs)} AOs)")
    aos = build_aos(atoms, templates)
    n = len(aos)
    print(f"Built {n} AOs")

    from lcao_wannier.parser import parse_overlap_and_fock_matrices_streaming
    _, S_R, _, _ = parse_overlap_and_fock_matrices_streaming(path, promote_complex='auto')
    Sref = np.real(S_R[(0, 0, 0)])
    Sref = Sref + Sref.T - np.diag(np.diag(Sref))   # symmetrize lower-tri storage

    S0 = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S0[i, j] = ao_overlap(aos[i], aos[j], np.zeros(3))
            S0[j, i] = S0[i, j]
    diff = np.abs(S0 - Sref)
    print(f"\nFull S(R=0) reconstruction vs reference:")
    print(f"  max abs diff: {diff.max():.3e}  at {np.unravel_index(diff.argmax(), diff.shape)}")
    print(f"  rms diff:     {np.sqrt((diff**2).mean()):.3e}")
    # block-wise diagnostics
    for lo, hi, name in [(0, 4, 'B s'), (4, 13, 'B p'), (13, 18, 'B d'),
                         (18, 36, 'atom1-atom2 block')]:
        if hi <= n:
            print(f"  block {name:18s} max diff: {diff[lo:hi, :].max():.3e}")

    # --- off-site S(g): validate against parsed S(R) for the nearest cells ---
    _, _, latt, _ = parse_overlap_and_fock_matrices_streaming(path, promote_complex='auto')
    latt = np.asarray(latt, float)
    # detect units: AO centres are Bohr; if latt looks like Angstrom, convert
    ANG2BOHR = 1.8897259886
    if np.max(np.abs(latt)) < 12:      # heuristic: Angstrom magnitude
        latt_bohr = latt * ANG2BOHR
    else:
        latt_bohr = latt
    print(f"\nLattice (Bohr) rows:\n{np.array2string(latt_bohr, precision=3)}")
    # pick a few nonzero cells with largest reference norm
    cells = sorted([g for g in S_R if g != (0, 0, 0)],
                   key=lambda g: -np.abs(S_R[g]).max())[:3]
    print("Off-site S(g) reconstruction (brute-force frame convention):")
    g = cells[0]
    n_arr = np.array(g, float)
    Sg_ref = np.real(S_R[g])
    convs = {
        'rows  n@latt': n_arr @ latt_bohr,
        'cols  latt@n': latt_bohr @ n_arr,
        '-rows':       -(n_arr @ latt_bohr),
        '-cols':       -(latt_bohr @ n_arr),
    }
    for name, Rg in convs.items():
        Sg = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Sg[i, j] = ao_overlap(aos[i], aos[j], Rg)
        d1 = np.abs(Sg - Sg_ref).max()
        d2 = np.abs(Sg.T - Sg_ref).max()
        print(f"  {name:14s} Rg={np.array2string(Rg,precision=2)}: "
              f"max|S-ref|={d1:.2e} max|S^T-ref|={d2:.2e}")
    print(f"  (cell {g}, refmax={np.abs(Sg_ref).max():.3e})")
    print(f"  storage check cell {g}: ref[0,18]={Sg_ref[0,18]:.4f} "
          f"ref[18,0]={Sg_ref[18,0]:.4f} (one ~0 => lower-tri only)")
    Rg = n_arr @ latt_bohr
    Smine = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Smine[i, j] = ao_overlap(aos[i], aos[j], Rg)
    print("  atom1-s x atom2-s block (rows=B s, cols=N s):")
    print("   mine:"); print(np.array2string(Smine[0:4, 18:22], precision=4, suppress_small=True))
    print("   ref :"); print(np.array2string(Sg_ref[0:4, 18:22], precision=4, suppress_small=True))
    print("   ref^T-block (cols->rows):"); print(np.array2string(Sg_ref[18:22, 0:4].T, precision=4, suppress_small=True))
    # The parser keeps only the LOWER triangle for off-site cells (S(g) is not
    # symmetric; the upper part lives in cell -g). So validate lower triangles.
    print("\n  LOWER-triangle validation (mine vs parser, all top cells):")
    for gg in cells:
        Rg = np.array(gg, float) @ latt_bohr
        Sg = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):       # lower triangle only
                Sg[i, j] = ao_overlap(aos[i], aos[j], Rg)
        ref = np.real(S_R[gg])
        d = np.abs(np.tril(Sg) - np.tril(ref)).max()
        print(f"    cell {gg}: max|tril(mine) - tril(ref)| = {d:.3e}  "
              f"(refmax={np.abs(np.tril(ref)).max():.3e})")

    # --- momentum-shifted primitive: b=0 must reduce to plain overlap ---
    A = np.array([0.1, -0.2, 0.3]); B = np.array([0.4, 0.5, -0.1])
    s0 = _prim_cart(1.2, A, (1, 0, 0), 0.8, B, (0, 1, 1))
    sm = _prim_cart_mom(1.2, A, (1, 0, 0), 0.8, B, (0, 1, 1), np.zeros(3))
    print(f"\nmomentum b=0 reduction: |plain - mom(b=0)| = {abs(s0 - sm):.3e}")


if __name__ == '__main__':
    main()
