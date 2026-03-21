"""
Virtual imaging computations: CoM, iCOM, and helper utilities.

icom_from_com() is the drop-in replacement for py4DSTEM's DPC phase
reconstruction (max_iter=1, step_size=1), implemented via Fourier
integration of the divergence of the centre-of-mass field.
"""

from __future__ import annotations

import numpy as np


def com_from_datacube(
    data: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the centre-of-mass shift field over all scan positions.

    Parameters
    ----------
    data  : (R_Nx, R_Ny, Q_Nx, Q_Ny) float array
    mask  : (Q_Nx, Q_Ny) bool/float array — the detector aperture

    Returns
    -------
    CoMx, CoMy : (R_Nx, R_Ny) arrays, mean-subtracted
    """
    R_Nx, R_Ny, Q_Nx, Q_Ny = data.shape
    mask_f = mask.astype(np.float32)

    ry_coord, rx_coord = np.meshgrid(
        np.arange(Q_Ny), np.arange(Q_Nx), indexing="ij"
    )
    # meshgrid gives (Q_Ny, Q_Nx) — transpose to match (Q_Nx, Q_Ny)
    rx_coord = rx_coord.T
    ry_coord = ry_coord.T

    CoMx = np.zeros((R_Nx, R_Ny), dtype=np.float32)
    CoMy = np.zeros((R_Nx, R_Ny), dtype=np.float32)

    for rx in range(R_Nx):
        for ry in range(R_Ny):
            ar = data[rx, ry] * mask_f
            tot = ar.sum()
            if tot > 0:
                CoMx[rx, ry] = (rx_coord * ar).sum() / tot
                CoMy[rx, ry] = (ry_coord * ar).sum() / tot

    CoMx -= CoMx.mean()
    CoMy -= CoMy.mean()
    return CoMx, CoMy


def icom_from_com(
    CoMx: np.ndarray,
    CoMy: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Reconstruct the iCOM phase from centre-of-mass fields.

    Uses Fourier-domain integration (identical to py4DSTEM DPC with
    max_iter=1, step_size=1):

        phi = IFFT( (kx·FFT(CoMx) + ky·FFT(CoMy)) / (kx² + ky²) )

    This works because CoM ≈ ∇φ, so FFT(CoMx) ≈ kx·FFT(φ).

    Parameters
    ----------
    CoMx, CoMy : (R_Nx, R_Ny) arrays — mean-subtracted CoM fields
    eps        : regularisation to avoid division by zero at DC

    Returns
    -------
    phase : (R_Nx, R_Ny) float array
    """
    Nx, Ny = CoMx.shape

    kx = np.fft.fftfreq(Nx)          # (Nx,)
    ky = np.fft.fftfreq(Ny)          # (Ny,)
    kx_g, ky_g = np.meshgrid(kx, ky, indexing="ij")   # (Nx, Ny)

    k2 = kx_g ** 2 + ky_g ** 2
    k2[0, 0] = 1.0  # avoid div/0; DC set to zero below

    phase_f = (
        kx_g * np.fft.fft2(CoMx.astype(np.float64))
        + ky_g * np.fft.fft2(CoMy.astype(np.float64))
    ) / (k2 + eps)

    phase_f[0, 0] = 0.0  # zero DC (arbitrary global offset)

    phase = np.real(np.fft.ifft2(phase_f)).astype(np.float32)
    return phase
