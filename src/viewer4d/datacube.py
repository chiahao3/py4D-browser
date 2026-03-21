"""
Thin 4D-STEM data container — replaces py4DSTEM.DataCube.

Holds a 4D array of shape (R_Nx, R_Ny, Q_Nx, Q_Ny) plus calibration
metadata. No heavy dependencies; accepts any array-like (numpy or torch).
"""

from __future__ import annotations

import numpy as np


class Calibration:
    """Simple calibration metadata: real-space and reciprocal-space pixel sizes."""

    def __init__(self):
        self._R_pixel_size: float = 1.0
        self._R_pixel_units: str = "pixels"
        self._Q_pixel_size: float = 1.0
        self._Q_pixel_units: str = "pixels"

    # ---- real space ----
    def get_R_pixel_size(self) -> float:
        return self._R_pixel_size

    def set_R_pixel_size(self, value: float):
        self._R_pixel_size = float(value)

    def get_R_pixel_units(self) -> str:
        return self._R_pixel_units

    def set_R_pixel_units(self, value: str):
        self._R_pixel_units = str(value)

    # ---- reciprocal space ----
    def get_Q_pixel_size(self) -> float:
        return self._Q_pixel_size

    def set_Q_pixel_size(self, value: float):
        self._Q_pixel_size = float(value)

    def get_Q_pixel_units(self) -> str:
        return self._Q_pixel_units

    def set_Q_pixel_units(self, value: str):
        self._Q_pixel_units = str(value)

    def __repr__(self) -> str:
        return (
            f"Calibration(\n"
            f"  R: {self._R_pixel_size} {self._R_pixel_units}/px\n"
            f"  Q: {self._Q_pixel_size} {self._Q_pixel_units}/px\n"
            f")"
        )


class DataCube:
    """
    Minimal 4D-STEM data container.

    Parameters
    ----------
    data : array-like, shape (R_Nx, R_Ny, Q_Nx, Q_Ny)
        The 4D dataset.  Can be a numpy ndarray, h5py Dataset (mmap),
        or a torch Tensor.
    calibration : Calibration, optional
        If omitted a default (1 px/px) calibration is created.

    Attributes
    ----------
    data        : the underlying array
    R_Nx, R_Ny  : scan dimensions
    Q_Nx, Q_Ny  : diffraction dimensions
    Rshape      : (R_Nx, R_Ny)
    Qshape      : (Q_Nx, Q_Ny)
    shape       : full 4-tuple
    calibration : Calibration object
    """

    def __init__(self, data, calibration: Calibration | None = None):
        if hasattr(data, "shape") and len(data.shape) != 4:
            raise ValueError(
                f"DataCube requires a 4D array, got shape {data.shape}"
            )
        self.data = data
        self.calibration = calibration if calibration is not None else Calibration()

    # ---- shape helpers ----
    @property
    def R_Nx(self) -> int:
        return self.data.shape[0]

    @property
    def R_Ny(self) -> int:
        return self.data.shape[1]

    @property
    def Q_Nx(self) -> int:
        return self.data.shape[2]

    @property
    def Q_Ny(self) -> int:
        return self.data.shape[3]

    @property
    def Rshape(self) -> tuple[int, int]:
        return (self.R_Nx, self.R_Ny)

    @property
    def Qshape(self) -> tuple[int, int]:
        return (self.Q_Nx, self.Q_Ny)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.data.shape

    def to_numpy(self) -> np.ndarray:
        """Return data as a numpy array (copies from device if needed)."""
        try:
            return self.data.numpy()  # torch tensor on CPU
        except AttributeError:
            return np.asarray(self.data)

    def __repr__(self) -> str:
        try:
            dtype = self.data.dtype
        except AttributeError:
            dtype = "unknown"
        return (
            f"DataCube(shape={self.data.shape}, dtype={dtype})\n"
            f"{self.calibration}"
        )
