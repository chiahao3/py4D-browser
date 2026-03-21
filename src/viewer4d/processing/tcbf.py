"""
Tilt-Corrected Bright-Field (tcBF) reconstruction.

Two implementations are provided:
  * tcbf_numpy  — CPU, pure NumPy.  Reference implementation, always available.
  * tcbf_torch  — GPU-accelerated via PyTorch.  Falls back to CPU if no CUDA.

Algorithm
---------
For each pixel (mx, my) inside the BF detector mask, the corresponding
real-space image I[..., mx, my] is shifted by (Sx[mx,my], Sy[mx,my]) scan
pixels in Fourier space (phase-ramp multiplication), then all shifted images
are accumulated into a single reconstruction.

The shift field is supplied by the caller (computed from detector geometry
in the GUI plugin) so that both automatic and manual workflows share the
same backend.

GPU speedup comes from batching *all* detector pixels simultaneously:
  1.  Extract all N active images  →  shape (N, R_Nx, R_Ny)
  2.  FFT the batch in one call
  3.  Multiply each FFT by its pre-computed phase ramp  →  shape (N, R_Nx, R_Ny)
  4.  IFFT + sum along N
This avoids Python-level loops over detector pixels entirely.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# NumPy (CPU) implementation — always available
# ---------------------------------------------------------------------------

def tcbf_numpy(
    data: np.ndarray,
    mask: np.ndarray,
    shifts_x: np.ndarray,
    shifts_y: np.ndarray,
    pad: bool = False,
    progress_callback=None,
) -> np.ndarray:
    """
    Tilt-corrected BF reconstruction on CPU.

    Parameters
    ----------
    data      : (R_Nx, R_Ny, Q_Nx, Q_Ny) array
    mask      : (Q_Nx, Q_Ny) bool array — BF detector aperture
    shifts_x  : (Q_Nx, Q_Ny) float array — scan-pixel shifts along x
    shifts_y  : (Q_Nx, Q_Ny) float array — scan-pixel shifts along y
    pad       : if True, pad the output to avoid wrap-around artefacts
    progress_callback : optional callable(int, int) for progress updates

    Returns
    -------
    reconstruction : (R_Nx, R_Ny) float array
    """
    R_Nx, R_Ny = data.shape[:2]
    pad_width = (
        int(np.maximum(np.abs(shifts_x).max(), np.abs(shifts_y).max()))
        if pad else 0
    )
    out_shape = (R_Nx + 2 * pad_width, R_Ny + 2 * pad_width)
    reconstruction = np.zeros(out_shape, dtype=np.float64)

    qx = np.fft.fftfreq(out_shape[0])
    qy = np.fft.fftfreq(out_shape[1])
    qx_g, qy_g = np.meshgrid(qx, qy, indexing="ij")
    qx_op = qx_g * (-2.0j * np.pi)
    qy_op = qy_g * (-2.0j * np.pi)

    active = np.argwhere(mask)
    N = len(active)

    for i, (mx, my) in enumerate(tqdm(active, desc="tcBF (CPU)", unit="px")):
        img_raw = data[:, :, mx, my].astype(np.float64)

        if pad:
            img = np.zeros(out_shape)
            img[:R_Nx, :R_Ny] = img_raw
        else:
            img = img_raw

        sx = float(shifts_x[mx, my])
        sy = float(shifts_y[mx, my])

        reconstruction += np.real(
            np.fft.ifft2(
                np.fft.fft2(img) * np.exp(qx_op * sx + qy_op * sy)
            )
        )

        if progress_callback is not None:
            progress_callback(i + 1, N)

    if pad:
        reconstruction = reconstruction[pad_width:-pad_width, pad_width:-pad_width]

    return reconstruction.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch (GPU) implementation
# ---------------------------------------------------------------------------

def tcbf_torch(
    data: np.ndarray,
    mask: np.ndarray,
    shifts_x: np.ndarray,
    shifts_y: np.ndarray,
    pad: bool = False,
    device: str | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    GPU-accelerated tilt-corrected BF reconstruction.

    Batches detector pixels together for a single FFT call per batch,
    giving 10-100× speedup over the numpy loop on a modern GPU.

    Parameters
    ----------
    data       : (R_Nx, R_Ny, Q_Nx, Q_Ny) numpy array
    mask       : (Q_Nx, Q_Ny) bool array
    shifts_x   : (Q_Nx, Q_Ny) float array (scan pixels)
    shifts_y   : (Q_Nx, Q_Ny) float array (scan pixels)
    pad        : pad output to suppress wrap-around artefacts
    device     : 'cuda', 'mps', 'cpu', or None (auto-select)
    batch_size : number of detector pixels processed per batch
                 (reduce if GPU memory is tight)

    Returns
    -------
    reconstruction : (R_Nx, R_Ny) float32 numpy array
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not found — falling back to NumPy tcBF.")
        return tcbf_numpy(data, mask, shifts_x, shifts_y, pad=pad)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"tcBF running on: {device}")
    dev = torch.device(device)

    R_Nx, R_Ny = data.shape[:2]
    pad_width = (
        int(np.maximum(np.abs(shifts_x).max(), np.abs(shifts_y).max()))
        if pad else 0
    )
    H = R_Nx + 2 * pad_width
    W = R_Ny + 2 * pad_width

    # Frequency grids  (H, W)
    qx = torch.fft.fftfreq(H, device=dev)
    qy = torch.fft.fftfreq(W, device=dev)
    qx_g, qy_g = torch.meshgrid(qx, qy, indexing="ij")   # (H, W)

    # Accumulator
    reconstruction = torch.zeros(H, W, device=dev, dtype=torch.float32)

    active = np.argwhere(mask)   # (N, 2)
    N = len(active)

    for start in tqdm(range(0, N, batch_size), desc="tcBF (GPU)", unit="batch"):
        batch_idx = active[start : start + batch_size]   # (B, 2)
        B = len(batch_idx)

        mx = batch_idx[:, 0]
        my = batch_idx[:, 1]

        # Extract images: (B, R_Nx, R_Ny)
        imgs_np = data[:, :, mx, my]            # (R_Nx, R_Ny, B)
        imgs_np = imgs_np.transpose(2, 0, 1)    # (B, R_Nx, R_Ny)
        imgs = torch.from_numpy(imgs_np.astype(np.float32)).to(dev)

        if pad:
            imgs_padded = torch.zeros(B, H, W, device=dev)
            imgs_padded[:, :R_Nx, :R_Ny] = imgs
            imgs = imgs_padded

        # Shifts: (B,)
        sx = torch.tensor(shifts_x[mx, my], dtype=torch.float32, device=dev)
        sy = torch.tensor(shifts_y[mx, my], dtype=torch.float32, device=dev)

        # Phase ramps: (B, H, W)
        phase_ramps = torch.exp(
            -2j * torch.pi * (
                sx[:, None, None] * qx_g[None]
                + sy[:, None, None] * qy_g[None]
            )
        )

        # Batch FFT → apply ramp → IFFT → sum
        imgs_f = torch.fft.fft2(imgs.to(torch.complex64))
        shifted = torch.fft.ifft2(imgs_f * phase_ramps).real   # (B, H, W)
        reconstruction += shifted.sum(dim=0)

        del imgs, imgs_f, shifted, phase_ramps

    if pad and pad_width > 0:
        reconstruction = reconstruction[pad_width:-pad_width, pad_width:-pad_width]

    return reconstruction.cpu().numpy()


# ---------------------------------------------------------------------------
# Shift field computation (shared by GUI plugins)
# ---------------------------------------------------------------------------

def compute_shift_field(
    mask: np.ndarray,
    max_shift: float,
    rotation_rad: float = 0.0,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-detector-pixel shift fields for tcBF.

    Maps each pixel inside *mask* to a scan-pixel shift proportional to
    its distance from the mask centre, scaled so the edge of the mask
    corresponds to *max_shift* pixels.

    Parameters
    ----------
    mask         : (Q_Nx, Q_Ny) bool array — BF detector aperture
    max_shift    : maximum shift in scan pixels (at the mask edge)
    rotation_rad : scan↔diffraction rotation angle in radians
    transpose    : if True, swap x and y before rotation

    Returns
    -------
    shifts_x, shifts_y : (Q_Nx, Q_Ny) float arrays
    """
    Q_Nx, Q_Ny = mask.shape
    x, y = np.meshgrid(np.arange(Q_Nx), np.arange(Q_Ny), indexing="ij")

    mask_comx = np.sum(mask * x) / np.sum(mask)
    mask_comy = np.sum(mask * y) / np.sum(mask)

    px = x - mask_comx
    py = y - mask_comy

    q_pix = np.hypot(px, py)
    scale = max_shift / np.max(q_pix * mask)
    shifts_x = px * scale
    shifts_y = py * scale

    R = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad),  np.cos(rotation_rad)],
    ])
    T = np.array([[0.0, 1.0], [1.0, 0.0]])
    if transpose:
        R = T @ R

    shifts = np.stack([shifts_x, shifts_y], axis=-1) @ R   # (Q_Nx, Q_Ny, 2)
    return shifts[..., 0], shifts[..., 1]
