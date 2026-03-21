"""
File writers for 4D-STEM data.
"""

from __future__ import annotations

import numpy as np
import h5py

from viewer4d.datacube import DataCube


def save_hdf5(filepath: str, datacube: DataCube) -> None:
    """
    Save a DataCube to HDF5 with calibration metadata.

    Layout
    ------
    /data           : float32 dataset, shape (R_Nx, R_Ny, Q_Nx, Q_Ny)
    /calibration/   : scalar datasets for pixel sizes and units
    """
    with h5py.File(filepath, "w") as f:
        data = np.asarray(datacube.data).astype(np.float32)
        f.create_dataset("data", data=data, compression="gzip", compression_opts=4)

        cal_grp = f.create_group("calibration")
        cal = datacube.calibration
        cal_grp.create_dataset("R_pixel_size",  data=cal.get_R_pixel_size())
        cal_grp.create_dataset("R_pixel_units", data=cal.get_R_pixel_units())
        cal_grp.create_dataset("Q_pixel_size",  data=cal.get_Q_pixel_size())
        cal_grp.create_dataset("Q_pixel_units", data=cal.get_Q_pixel_units())

    print(f"Saved DataCube to {filepath}")


def save_raw_float32(filepath: str, datacube: DataCube) -> None:
    """Save raw float32 binary (no header — shape information is lost)."""
    np.asarray(datacube.data).astype(np.float32).tofile(filepath)
    print(f"Saved raw float32 to {filepath}")
