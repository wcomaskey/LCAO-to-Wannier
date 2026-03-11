"""
Orbital Rotation Matrices for Real Spherical Harmonics

Computes how atomic orbital basis functions transform under point group
rotations. Uses numerical evaluation (test-point + lstsq fitting) instead
of symbolic algebra, eliminating the SymPy dependency.

Orbital orderings follow Wannier90 convention:
    s:   [s]                                       (1 function)
    p:   [pz, px, py]                              (3 functions)
    d:   [dz2, dxz, dyz, dx2-y2, dxy]            (5 functions)
    f:   [fz3, fxz2, fyz2, fz(x2-y2), fxyz,      (7 functions)
          fx(x2-3y2), fy(3x2-y2)]
    t2g: [dxz, dyz, dxy]                          (3 functions, subset of d)
    pz:  [pz]                                      (1 function, subset of p)

Mathematical basis:
    For a rotation R, the orbital transformation is:
        P_R f(r) = f(R^{-1} r)
    The representation matrix D[m,n] satisfies:
        phi_n(R^{-1}r) = sum_m D[m,n] phi_m(r)

Reference: Section 3 of wannhr_symm (Changming Yue, arxiv:1805.12148)
"""

import numpy as np
from typing import Tuple, List


# ============================================================================
# Orbital evaluation functions
# ============================================================================

def _s_orbital(x: float, y: float, z: float) -> np.ndarray:
    """Evaluate s-orbital at (x,y,z). Returns array of shape (1,)."""
    return np.array([1.0])


def _p_orbitals(x: float, y: float, z: float) -> np.ndarray:
    """Evaluate p-orbitals at (x,y,z) in Wannier90 order [pz, px, py]."""
    return np.array([z, x, y])


def _d_orbitals(x: float, y: float, z: float) -> np.ndarray:
    """
    Evaluate d-orbitals at (x,y,z) in Wannier90 order:
    [dz2, dxz, dyz, dx2-y2, dxy]
    """
    sq3 = np.sqrt(3.0)
    return np.array([
        (2*z*z - x*x - y*y) / (2*sq3),  # dz2
        x*z,                              # dxz
        y*z,                              # dyz
        (x*x - y*y) / 2.0,              # dx2-y2
        x*y,                              # dxy
    ])


def _f_orbitals(x: float, y: float, z: float) -> np.ndarray:
    """
    Evaluate f-orbitals at (x,y,z) in Wannier90 order:
    [fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)]
    """
    sq15 = np.sqrt(15.0)
    sq10 = np.sqrt(10.0)
    sq6 = np.sqrt(6.0)
    return np.array([
        z*(2*z*z - 3*x*x - 3*y*y) / (2*sq15),    # fz3
        x*(4*z*z - x*x - y*y) / (2*sq10),          # fxz2
        y*(4*z*z - x*x - y*y) / (2*sq10),          # fyz2
        z*(x*x - y*y) / 2.0,                        # fz(x2-y2)
        x*y*z,                                       # fxyz
        x*(x*x - 3*y*y) / (2*sq6),                  # fx(x2-3y2)
        y*(3*x*x - y*y) / (2*sq6),                  # fy(3x2-y2)
    ])


# ============================================================================
# Test point generation for lstsq fitting
# ============================================================================

def _generate_test_points_quadratic() -> List[np.ndarray]:
    """Generate test points on unit sphere for fitting quadratic (d-orbital) coefficients."""
    pts = []
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for x in vals:
        for y in vals:
            for z in vals:
                r = np.sqrt(x*x + y*y + z*z)
                if r > 0.1:
                    pts.append(np.array([x, y, z]) / r)

    # Add icosahedral vertices for numerical stability
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            v = np.array([0, s1, s2*phi])
            pts.append(v / np.linalg.norm(v))
            v = np.array([s1, s2*phi, 0])
            pts.append(v / np.linalg.norm(v))
            v = np.array([s2*phi, 0, s1])
            pts.append(v / np.linalg.norm(v))
    return pts


def _generate_test_points_cubic() -> List[np.ndarray]:
    """Generate test points for fitting cubic (f-orbital) coefficients."""
    pts = _generate_test_points_quadratic()
    for theta in np.linspace(0.1, np.pi - 0.1, 8):
        for phi_angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            z = np.cos(theta)
            pts.append(np.array([x, y, z]))
    return pts


# ============================================================================
# Rotation matrix computation
# ============================================================================

def orbital_rotation_s(R_cart: np.ndarray) -> np.ndarray:
    """s-orbital rotation matrix: always [[1]].

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Cartesian rotation matrix

    Returns
    -------
    ndarray (1, 1)
    """
    return np.array([[1.0]])


def orbital_rotation_p(R_cart: np.ndarray) -> np.ndarray:
    """
    p-orbital rotation matrix in Wannier90 basis [pz, px, py].

    For p-orbitals, the representation matrix is simply the rotation matrix
    in the [z, x, y] reordering.

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Cartesian rotation matrix

    Returns
    -------
    ndarray (3, 3)
    """
    # Cartesian order: x=0, y=1, z=2
    # Wannier90 order: pz=0, px=1, py=2
    # Map: pz->z(2), px->x(0), py->y(1)
    idx = [2, 0, 1]  # [z, x, y] indices in Cartesian
    D = R_cart[np.ix_(idx, idx)]
    return D


def orbital_rotation_d(R_cart: np.ndarray) -> np.ndarray:
    """
    d-orbital rotation matrix in Wannier90 basis [dz2, dxz, dyz, dx2-y2, dxy].

    Uses numerical test-point evaluation + least-squares fitting.

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Cartesian rotation matrix

    Returns
    -------
    ndarray (5, 5)
    """
    R_inv = np.linalg.inv(R_cart)
    D = np.zeros((5, 5))

    test_points = _generate_test_points_quadratic()
    B = np.zeros((len(test_points), 5))
    for ip, p in enumerate(test_points):
        B[ip] = _d_orbitals(p[0], p[1], p[2])

    for j in range(5):
        vals = np.zeros(len(test_points))
        for ip, p in enumerate(test_points):
            rp = R_inv @ p
            vals[ip] = _d_orbitals(rp[0], rp[1], rp[2])[j]
        D[:, j], _, _, _ = np.linalg.lstsq(B, vals, rcond=None)

    D[np.abs(D) < 1e-10] = 0.0
    return D


def orbital_rotation_f(R_cart: np.ndarray) -> np.ndarray:
    """
    f-orbital rotation matrix in Wannier90 basis:
    [fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)]

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Cartesian rotation matrix

    Returns
    -------
    ndarray (7, 7)
    """
    R_inv = np.linalg.inv(R_cart)
    D = np.zeros((7, 7))

    test_points = _generate_test_points_cubic()
    B = np.zeros((len(test_points), 7))
    for ip, p in enumerate(test_points):
        B[ip] = _f_orbitals(p[0], p[1], p[2])

    for j in range(7):
        vals = np.zeros(len(test_points))
        for ip, p in enumerate(test_points):
            rp = R_inv @ p
            vals[ip] = _f_orbitals(rp[0], rp[1], rp[2])[j]
        D[:, j], _, _, _ = np.linalg.lstsq(B, vals, rcond=None)

    D[np.abs(D) < 1e-10] = 0.0
    return D


def orbital_rotation_t2g(R_cart: np.ndarray) -> np.ndarray:
    """
    t2g orbital rotation matrix: 3×3 submatrix of d-orbital rotation
    for the [dxz, dyz, dxy] subset.

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Cartesian rotation matrix

    Returns
    -------
    ndarray (3, 3)
    """
    D_d = orbital_rotation_d(R_cart)
    # d-orbital ordering: [dz2(0), dxz(1), dyz(2), dx2-y2(3), dxy(4)]
    # t2g subset: dxz(1), dyz(2), dxy(4)
    idx = [1, 2, 4]
    return D_d[np.ix_(idx, idx)]


def orbital_rotation_pz(R_cart: np.ndarray) -> np.ndarray:
    """
    pz orbital rotation matrix: 1×1 element of p-orbital rotation
    for the pz component only.

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Cartesian rotation matrix

    Returns
    -------
    ndarray (1, 1)
    """
    D_p = orbital_rotation_p(R_cart)
    # p-orbital ordering: [pz(0), px(1), py(2)]
    return D_p[0:1, 0:1]


# ============================================================================
# Dispatcher
# ============================================================================

# Orbital dimensions
ORBITAL_DIMS = {
    's': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9,
    't2g': 3, 'pz': 1,
}


def get_orbital_dim(orbital_type: str) -> int:
    """Return the number of m_l basis functions for an orbital type."""
    key = orbital_type.lower()
    if key not in ORBITAL_DIMS:
        raise ValueError(
            f"Unsupported orbital type: '{orbital_type}'. "
            f"Supported types: {list(ORBITAL_DIMS.keys())}"
        )
    return ORBITAL_DIMS[key]


def get_orbital_rotation(orbital_type: str, R_cart: np.ndarray) -> np.ndarray:
    """
    Get the orbital rotation matrix for a given orbital type and rotation.

    Parameters
    ----------
    orbital_type : str
        One of 's', 'p', 'd', 'f', 't2g', 'pz'
    R_cart : ndarray (3, 3)
        Rotation matrix in Cartesian coordinates

    Returns
    -------
    ndarray
        Orbital rotation matrix D
    """
    key = orbital_type.lower()
    if key == 's':
        return orbital_rotation_s(R_cart)
    elif key == 'p':
        return orbital_rotation_p(R_cart)
    elif key == 'd':
        return orbital_rotation_d(R_cart)
    elif key == 'f':
        return orbital_rotation_f(R_cart)
    elif key == 't2g':
        return orbital_rotation_t2g(R_cart)
    elif key == 'pz':
        return orbital_rotation_pz(R_cart)
    else:
        raise ValueError(
            f"Unsupported orbital type: '{orbital_type}'. "
            f"Supported types: {list(ORBITAL_DIMS.keys())}"
        )
