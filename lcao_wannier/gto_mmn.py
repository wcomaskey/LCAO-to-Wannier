"""Exact analytic GTO overlaps and MMN for CRYSTAL LCAO output.

Computes the Wannier90 MMN overlaps M_mn(k,b) = <u_mk|u_n,k+b> exactly from the
Gaussian basis, with no midpoint or Löwdin approximation:

    M(k,b) = C(k)^dag . Stilde^{(b)}(k+b) . C(k+b)
    Stilde^{(b)}_uv(k+b) = sum_g e^{i(k+b).g} S^{(b)}_uv(g)
    S^{(b)}_uv(g) = integral phi_u(r-A_u) e^{-i b.r} phi_v(r-A_v-g) dr

The momentum-shifted Gaussian integral is analytic: the e^{-i b.r} factor shifts
the Gaussian product centre to the complex point P - i b/2p and adds the scalar
e^{-i b.P} e^{-|b|^2/4p}. Reduces exactly to the plain overlap at b=0.

CRYSTAL conventions (validated to printed precision against S(R), see
scripts/analytic_mmn_dev.py):
  * Convention II Bloch phase e^{ik.g}; AOs centred at the nucleus.
  * printed coefficients used with UNNORMALIZED primitives e^{-a r^2}; the
    contracted AO is normalized to unit self-overlap.
  * order of internal storage (manual basisset p.27): S=s; P=x,y,z;
    D=2z^2-x^2-y^2,xz,yz,x^2-y^2,xy; F=7, G=9 real solid harmonics (see _FCART,
    _GCART). The Hermite/overlap engine is angular-momentum general; only the
    real-solid-harmonic -> Cartesian tables are per-l.
"""
import re
import numpy as np
from math import pi, sqrt

ANG2BOHR = 1.8897259886

# real-solid-harmonic -> Cartesian (powers, coefficient)
_DCART = [
    [((0, 0, 2), 2.0), ((2, 0, 0), -1.0), ((0, 2, 0), -1.0)],  # 2z^2-x^2-y^2
    [((1, 0, 1), 1.0)],                                         # xz
    [((0, 1, 1), 1.0)],                                         # yz
    [((2, 0, 0), 1.0), ((0, 2, 0), -1.0)],                      # x^2-y^2
    [((1, 1, 0), 1.0)],                                         # xy
]
_PCART = [[((1, 0, 0), 1.0)], [((0, 1, 0), 1.0)], [((0, 0, 1), 1.0)]]
_SCART = [[((0, 0, 0), 1.0)]]
# F real solid harmonics, CRYSTAL "order of internal storage" (manual basisset
# p.27): (2z^2-3x^2-3y^2)z, (4z^2-x^2-y^2)x, (4z^2-x^2-y^2)y, (x^2-y^2)z, xyz,
# (x^2-3y^2)x, (3x^2-y^2)y.
_FCART = [
    [((0, 0, 3), 2.0), ((2, 0, 1), -3.0), ((0, 2, 1), -3.0)],   # (2z^2-3x^2-3y^2)z
    [((1, 0, 2), 4.0), ((3, 0, 0), -1.0), ((1, 2, 0), -1.0)],   # (4z^2-x^2-y^2)x
    [((0, 1, 2), 4.0), ((2, 1, 0), -1.0), ((0, 3, 0), -1.0)],   # (4z^2-x^2-y^2)y
    [((2, 0, 1), 1.0), ((0, 2, 1), -1.0)],                      # (x^2-y^2)z
    [((1, 1, 1), 1.0)],                                         # xyz
    [((3, 0, 0), 1.0), ((1, 2, 0), -3.0)],                      # (x^2-3y^2)x
    [((2, 1, 0), 3.0), ((0, 3, 0), -1.0)],                      # (3x^2-y^2)y
]
# G real solid harmonics, same source (manual basisset p.27). G shells are
# polarization-only in CRYSTAL but are included here for completeness.
_GCART = [
    [((4, 0, 0), 3.0), ((2, 2, 0), 6.0), ((2, 0, 2), -24.0),
     ((0, 4, 0), 3.0), ((0, 2, 2), -24.0), ((0, 0, 4), 8.0)],   # 3x^4+6x^2y^2-24x^2z^2+3y^4-24y^2z^2+8z^4
    [((1, 0, 3), 4.0), ((3, 0, 1), -3.0), ((1, 2, 1), -3.0)],   # (4z^2-3x^2-3y^2)xz
    [((0, 1, 3), 4.0), ((2, 1, 1), -3.0), ((0, 3, 1), -3.0)],   # (4z^2-3x^2-3y^2)yz
    [((2, 0, 2), 6.0), ((4, 0, 0), -1.0), ((0, 2, 2), -6.0),
     ((0, 4, 0), 1.0)],                                         # (x^2-y^2)(6z^2-x^2-y^2)
    [((1, 1, 2), 6.0), ((3, 1, 0), -1.0), ((1, 3, 0), -1.0)],   # (6z^2-x^2-y^2)xy
    [((3, 0, 1), 1.0), ((1, 2, 1), -3.0)],                      # (x^2-3y^2)xz
    [((2, 1, 1), 3.0), ((0, 3, 1), -1.0)],                      # (3x^2-y^2)yz
    [((4, 0, 0), 1.0), ((2, 2, 0), -6.0), ((0, 4, 0), 1.0)],    # x^4-6x^2y^2+y^4
    [((3, 1, 0), 1.0), ((1, 3, 0), -1.0)],                      # (x^2-y^2)xy
]
_LCART = {0: _SCART, 1: _PCART, 2: _DCART, 3: _FCART, 4: _GCART}


# ----------------------------------------------------------------------------
# Basis parsing
# ----------------------------------------------------------------------------
def parse_gto_basis(lines):
    """Parse the 'LOCAL ATOMIC FUNCTIONS BASIS SET' block.

    Returns the ordered AO list matching the H(R)/S(R) matrix ordering. Each AO
    is a dict(center (Bohr), terms=[((lx,ly,lz), ang_coef)], prims=[(exp,coef)],
    norm). Handles atom basis reuse (atoms of equal symbol share a template)."""
    LMAP = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}
    COL = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3}
    start = next(i for i, ln in enumerate(lines)
                 if 'LOCAL ATOMIC FUNCTIONS BASIS SET' in ln)
    atoms = []           # (idx, sym, center)
    sym_shells = {}      # sym -> [shell dict] from the first atom of that sym
    cur_sym = cur_center = cur_shell = None
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
            cur_shell = dict(l=LMAP[m.group(3)], prims=[])
            first = next(a[0] for a in atoms if a[1] == cur_sym)
            if atoms[-1][0] == first:
                sym_shells.setdefault(cur_sym, []).append(cur_shell)
            continue
        m = prim_re.match(ln)
        if m and cur_shell is not None:
            v = [float(m.group(i)) for i in range(1, 5)]
            cur_shell['prims'].append((v[0], v[COL[cur_shell['l']]]))

    aos = []
    for idx, sym, center in atoms:
        for sh in sym_shells[sym]:
            for comp in _LCART[sh['l']]:
                aos.append(dict(center=center, terms=comp, prims=sh['prims']))
    for ao in aos:
        ao['norm'] = 1.0
        ao['norm'] = 1.0 / sqrt(_pair_overlap(ao, ao, np.zeros(3)).real)
    return aos


# ----------------------------------------------------------------------------
# Gaussian integrals
# ----------------------------------------------------------------------------
def _herm_1d(p, PA, PB, la, lb):
    """1-D overlap polynomial via Hermite recursion; PA/PB may be complex.

    p, PA, PB may be scalars or broadcastable arrays (e.g. one entry per
    primitive pair); the recursion is elementwise and returns S[la, lb] with the
    same shape as the broadcast of the inputs."""
    p = np.asarray(p, dtype=complex)
    PA = np.asarray(PA, dtype=complex)
    PB = np.asarray(PB, dtype=complex)
    shp = np.broadcast_shapes(p.shape, PA.shape, PB.shape)
    half = 0.5 / p
    S = np.zeros((la + 2, lb + 2) + shp, dtype=complex)
    S[0, 0] = np.ones(shp, dtype=complex)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                S[i, j] = PA * S[i - 1, j] + half * (
                    (i - 1) * S[i - 2, j] + j * S[i - 1, j - 1])
            else:
                S[i, j] = PB * S[i, j - 1] + half * (
                    i * S[i - 1, j - 1] + (j - 1) * S[i, j - 2])
    return S[la, lb]


def _prim_mom(a, A, lA, b, B, lB, bvec):
    """<g_a(r-A)| e^{-i bvec.r} | g_b(r-B)>, analytic (complex).

    a, b may be scalars or broadcastable arrays of exponents (one element per
    primitive pair); the return has the broadcast shape."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = a + b
    AB = A - B
    P = [(a * A[d] + b * B[d]) / p for d in range(3)]
    bdotP = bvec[0] * P[0] + bvec[1] * P[1] + bvec[2] * P[2]
    res = ((pi / p) ** 1.5
           * np.exp(-a * b / p * np.dot(AB, AB))
           * np.exp(-np.dot(bvec, bvec) / (4 * p))
           * np.exp(-1j * bdotP)).astype(complex)
    for d in range(3):
        Ppd = P[d] - 0.5j * bvec[d] / p
        res = res * _herm_1d(p, Ppd - A[d], Ppd - B[d], lA[d], lB[d])
    return res


def _pair_overlap(ao_i, ao_j, Rj, bvec=None):
    """<ao_i(0)| e^{-i bvec.r} | ao_j(+Rj)>. bvec=None -> plain overlap.

    Vectorized over the primitive-pair grid: the inner double loop over
    primitives is replaced by a single broadcast call to _prim_mom."""
    if bvec is None:
        bvec = np.zeros(3)
    Ci = ao_i['center']
    Cj = ao_j['center'] + Rj
    ea = np.array([pr[0] for pr in ao_i['prims']])[:, None]   # (na, 1)
    ca = np.array([pr[1] for pr in ao_i['prims']])[:, None]
    eb = np.array([pr[0] for pr in ao_j['prims']])[None, :]   # (1, nb)
    cb = np.array([pr[1] for pr in ao_j['prims']])[None, :]
    cab = ca * cb                                              # (na, nb)
    tot = 0.0 + 0.0j
    for (li, ci) in ao_i['terms']:
        for (lj, cj) in ao_j['terms']:
            block = _prim_mom(ea, Ci, li, eb, Cj, lj, bvec)   # (na, nb)
            tot += ci * cj * np.sum(cab * block)
    return tot * ao_i['norm'] * ao_j['norm']


# ----------------------------------------------------------------------------
# MMN assembly
# ----------------------------------------------------------------------------
def reciprocal_bohr(lattice_bohr):
    """Reciprocal lattice rows (1/Bohr), 2pi convention: b_i . a_j = 2pi d_ij."""
    return 2 * pi * np.linalg.inv(lattice_bohr).T


def build_Sb_cells(aos, bvec, cells_R, cutoff=None):
    """S^{(b)}(g) for every cell g. cells_R = [(g_tuple, R_cart_bohr)]. Returns
    dict g_tuple -> (n_ao,n_ao) complex. cutoff: skip AO pairs whose centres are
    farther than `cutoff` Bohr (overlap negligible)."""
    n = len(aos)
    centers = np.array([ao['center'] for ao in aos])
    out = {}
    for g, Rg in cells_R:
        M = np.zeros((n, n), dtype=complex)
        cj = centers + Rg
        for i in range(n):
            di = np.linalg.norm(cj - centers[i], axis=1)
            for j in range(n):
                if cutoff is not None and di[j] > cutoff:
                    continue
                M[i, j] = _pair_overlap(aos[i], aos[j], Rg, bvec)
        out[g] = M
    return out


def Sb_at_kpb(Sb_cells, kpb_frac):
    """Fourier sum sum_g e^{i 2pi (k+b).g} S^{(b)}(g)."""
    n = next(iter(Sb_cells.values())).shape[0]
    S = np.zeros((n, n), dtype=complex)
    for g, M in Sb_cells.items():
        S += np.exp(2j * pi * np.dot(kpb_frac, g)) * M
    return S


def mmn_block(C_k, C_kpb, Sb_cells, kpb_frac):
    """M_mn(k,b) = C(k)^dag . Stilde^{(b)}(k+b) . C(k+b)."""
    S = Sb_at_kpb(Sb_cells, kpb_frac)
    return C_k.conj().T @ S @ C_kpb
