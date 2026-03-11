"""
Euler Angle Extraction and Spinor D-matrix

Provides the spin-1/2 (spinor) representation matrices needed for SOC
systems. A rotation R in SO(3) is parameterized by ZYZ Euler angles
(alpha, beta, gamma), and the corresponding SU(2) matrix is the
Wigner D^{1/2} matrix.

Convention: R = Rz(gamma) @ Ry(beta) @ Rz(alpha)  (ZYZ)

Reference: lib/rotate.py and lib/get_euler_angle.py from wannhr_symm
"""

import numpy as np
from typing import Tuple


def rmat_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles (alpha, beta, gamma) from a 3×3 rotation matrix.

    Uses ZYZ convention: R = Rz(gamma) @ Ry(beta) @ Rz(alpha)

    Parameters
    ----------
    R : ndarray (3, 3)
        Rotation matrix (must be orthogonal, det = ±1)

    Returns
    -------
    alpha, beta, gamma : float
        Euler angles in radians.
        alpha, gamma in [0, 2π], beta in [0, π]
    """
    # Handle gimbal lock cases
    if abs(R[2, 2]) < 1.0:
        # General case
        beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
        sin_beta = np.sin(beta)

        cos_gamma = -R[2, 0] / sin_beta
        sin_gamma = R[2, 1] / sin_beta
        gamma = _angle_from_sincos(sin_gamma, cos_gamma)

        cos_alpha = R[0, 2] / sin_beta
        sin_alpha = R[1, 2] / sin_beta
        alpha = _angle_from_sincos(sin_alpha, cos_alpha)
    else:
        # beta = 0 or beta = pi (gimbal lock)
        if R[2, 2] > 0:
            # beta = 0: R = Rz(alpha + gamma)
            beta = 0.0
            gamma = 0.0
            alpha = np.arccos(np.clip(R[1, 1], -1.0, 1.0))
            if -R[0, 1] < 0.0:
                alpha = -alpha
        else:
            # beta = pi: R = Rz(alpha - gamma) @ diag(-1,-1,1)
            beta = np.pi
            gamma = 0.0
            alpha = np.arccos(np.clip(R[1, 1], -1.0, 1.0))
            if -R[0, 1] < 0.0:
                alpha = -alpha

    return alpha, beta, gamma


def _angle_from_sincos(sina: float, cosa: float) -> float:
    """Determine angle in [0, 2π] from sin and cos values."""
    cosa = np.clip(cosa, -1.0, 1.0)
    angle = np.arccos(cosa)
    if sina < 0.0:
        angle = 2.0 * np.pi - angle
    return angle


def spinor_dmatrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute the spin-1/2 Wigner D-matrix for Euler angles (ZYZ convention).

    D[0,0] =  exp(-i(α+γ)/2) cos(β/2)
    D[0,1] = -exp(-i(α-γ)/2) sin(β/2)
    D[1,0] =  exp(+i(α-γ)/2) sin(β/2)
    D[1,1] =  exp(+i(α+γ)/2) cos(β/2)

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (ZYZ convention)

    Returns
    -------
    ndarray (2, 2) complex128
        Spinor D-matrix
    """
    D = np.zeros((2, 2), dtype=np.complex128)
    D[0, 0] = np.exp(-1j * (alpha + gamma) / 2.0) * np.cos(beta / 2.0)
    D[0, 1] = -np.exp(-1j * (alpha - gamma) / 2.0) * np.sin(beta / 2.0)
    D[1, 0] = np.exp(1j * (alpha - gamma) / 2.0) * np.sin(beta / 2.0)
    D[1, 1] = np.exp(1j * (alpha + gamma) / 2.0) * np.cos(beta / 2.0)
    return D


def get_spinor_dmatrix(R_cart: np.ndarray) -> np.ndarray:
    """
    Get the spinor D-matrix for a Cartesian rotation matrix.

    For improper rotations (det = -1), uses the proper part (det * R)
    since the spinor representation is for SO(3), not O(3).

    Parameters
    ----------
    R_cart : ndarray (3, 3)
        Rotation matrix in Cartesian coordinates

    Returns
    -------
    ndarray (2, 2) complex128
        Spinor D^{1/2} matrix
    """
    det = np.linalg.det(R_cart)
    R_proper = R_cart * np.sign(det)  # Remove inversion if present
    alpha, beta, gamma = rmat_to_euler(R_proper)
    D = spinor_dmatrix(alpha, beta, gamma)

    # Clean small numerical noise
    for i in range(2):
        for j in range(2):
            if abs(D[i, j].real) < 1e-10:
                D[i, j] = 1j * D[i, j].imag
            if abs(D[i, j].imag) < 1e-10:
                D[i, j] = D[i, j].real + 0j
    return D
