"""
Menu action callbacks mixed into DataViewer via class-level imports.

File loading uses viewer4d.io.readers (rosettasciio + h5py + numpy).
File saving uses viewer4d.io.writers.
No py4DSTEM dependency.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from viewer4d.datacube import DataCube, Calibration
from viewer4d.io.readers import load_file as _load_file, _find_nd_datasets
from viewer4d.io.writers import save_hdf5, save_raw_float32
from viewer4d.help_menu import KeyboardMapMenu
from viewer4d.dialogs import ResizeDialog

import h5py

# Supported extensions for the open dialog
_OPEN_FILTER = (
    "4D-STEM Data "
    "(*.dm3 *.dm4 *.raw *.mib *.h5 *.hdf5 *.emd *.npy *.npz)"
    ";;Any file (*)"
)


# ---------------------------------------------------------------------------
# Load actions
# ---------------------------------------------------------------------------

def load_data_auto(self):
    filename = self.show_file_dialog()
    if filename:
        self.load_file(filename)


def load_data_mmap(self):
    filename = self.show_file_dialog()
    if filename:
        self.load_file(filename, mmap=True)


def load_data_bin(self):
    filename = self.show_file_dialog()
    if filename:
        self.load_file(filename, mmap=False, binning=4)


def load_data_arina(self):
    """Load Arina detector data (Dectris EIGER, typically stored as HDF5)."""
    filename = self.show_file_dialog()
    if not filename:
        return

    try:
        from rosettasciio.eiger import file_reader as eiger_read
        signals = eiger_read(filename)
    except Exception:
        # Fall through to generic HDF5 reader
        signals = None

    if signals:
        import numpy as np
        data = np.asarray(signals[0]["data"])
    else:
        # Try h5py directly
        with h5py.File(filename, "r") as f:
            dsets = _find_nd_datasets(f, N=4)
            if not dsets:
                dsets = _find_nd_datasets(f, N=3)
            if not dsets:
                self.statusBar().showMessage("No 4D/3D data found in file", 5_000)
                return
            data = dsets[0][()]

    if data.ndim == 3:
        # shape (N_frames, det_x, det_y) — need to infer scan shape
        N = data.shape[0]
        Nxy = int(np.round(np.sqrt(N)))
        if Nxy * Nxy == N:
            data = data.reshape(Nxy, Nxy, *data.shape[1:])
        else:
            self.statusBar().showMessage(
                f"Arina: {N} frames — not square, needs manual reshape", 5_000
            )
            new_shape = ResizeDialog.get_new_size([1, N], parent=self)
            data = data.reshape(*new_shape, *data.shape[1:])

    self.datacube = DataCube(data)
    self.update_scalebars()
    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)
    self.setWindowTitle(filename)


def load_file(self, filepath: str, mmap: bool = False, binning: int = 1):
    if not filepath:
        return
    print(f"Loading {filepath!r}  (mmap={mmap}, binning={binning})")

    ext = os.path.splitext(filepath)[-1].lower()

    # HDF5: intercept to support reshape dialog for 3D data
    if ext in (".h5", ".hdf5", ".emd"):
        with h5py.File(filepath, "r") as f:
            dsets4 = _find_nd_datasets(f, N=4)
            if dsets4:
                data = dsets4[0] if mmap else dsets4[0][()]
                from viewer4d.io.readers import _calibration_from_hdf5
                cal = _calibration_from_hdf5(dsets4[0])
                self.datacube = DataCube(data, calibration=cal)
            else:
                dsets3 = _find_nd_datasets(f, N=3)
                if dsets3:
                    arr = dsets3[0] if mmap else dsets3[0][()]
                    new_shape = ResizeDialog.get_new_size([1, arr.shape[0]], parent=self)
                    self.datacube = DataCube(arr.reshape(*new_shape, *arr.shape[1:]))
                else:
                    raise ValueError("No 4D (or 3D) data found in HDF5 file.")
    else:
        self.datacube = _load_file(filepath, mmap=mmap, binning=binning)

    self.update_scalebars()
    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)
    self.setWindowTitle(filepath)


def set_datacube(self, datacube: DataCube, window_title: str = ""):
    self.datacube = datacube
    self.update_scalebars()
    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)
    if window_title:
        self.setWindowTitle(window_title)


# ---------------------------------------------------------------------------
# Scalebar update
# ---------------------------------------------------------------------------

_REALSPACE_UNIT_MAP = {"A": "Å"}
_RECIP_UNIT_MAP = {"A^-1": "Å⁻¹"}


def update_scalebars(self):
    cal = self.datacube.calibration

    q_size = cal.get_Q_pixel_size()
    q_units = cal.get_Q_pixel_units()
    self.diffraction_scale_bar.pixel_size = q_size
    self.diffraction_scale_bar.units = _RECIP_UNIT_MAP.get(q_units, q_units)

    r_size = cal.get_R_pixel_size()
    r_units = cal.get_R_pixel_units()
    self.real_space_scale_bar.pixel_size = r_size
    self.real_space_scale_bar.units = _REALSPACE_UNIT_MAP.get(r_units, r_units)

    fft_size = 1.0 / r_size / self.datacube.R_Ny if r_size > 0 else 1.0
    self.fft_scale_bar.pixel_size = fft_size
    self.fft_scale_bar.units = f"{_REALSPACE_UNIT_MAP.get(r_units, r_units)}⁻¹"

    self.diffraction_scale_bar.updateBar()
    self.real_space_scale_bar.updateBar()
    self.fft_scale_bar.updateBar()


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

def reshape_data(self):
    if self.datacube is None:
        return
    new_shape = ResizeDialog.get_new_size(list(self.datacube.shape[:2]), parent=self)
    self.datacube.data = self.datacube.data.reshape(
        *new_shape, *self.datacube.data.shape[2:]
    )
    print(f"Reshaped data to {new_shape}")
    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_datacube(self, save_format: str):
    assert save_format in ["Raw float32", "HDF5"], f"Unknown format: {save_format}"
    assert self.datacube is not None, "No datacube loaded."

    if save_format == "Raw float32":
        response = QMessageBox.question(
            self,
            "Save RAW file?",
            (
                "Raw binary files encode no shape, endianness, or ordering "
                "information. Saving to HDF5 is strongly recommended.\n\n"
                "Continue saving as RAW?"
            ),
            QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if response == QMessageBox.Cancel:
            self.statusBar().showMessage("Cancelled.", 3_000)
            return

    filename = self.get_savefile_name(save_format)
    if not filename:
        return

    if save_format == "Raw float32":
        save_raw_float32(filename, self.datacube)
    elif save_format == "HDF5":
        save_hdf5(filename, self.datacube)

    self.statusBar().showMessage(f"Saved to {filename}", 5_000)


def export_virtual_image(self, im_format: str, im_type: str):
    assert im_type in ["image", "diffraction"]

    filename = self.get_savefile_name(im_format)
    if not filename:
        return

    view = (
        self.real_space_widget if im_type == "image" else self.diffraction_space_widget
    )
    vimg = view.image.T
    vmin, vmax = view.getLevels()

    if im_format == "PNG (display)":
        plt.imsave(fname=filename, arr=vimg, vmin=vmin, vmax=vmax, format="png", cmap="gray")
    elif im_format == "TIFF (display)":
        plt.imsave(fname=filename, arr=vimg, vmin=vmin, vmax=vmax, format="tiff", cmap="gray")
    elif im_format == "TIFF (raw)":
        from tifffile import TiffWriter
        raw = (
            self.unscaled_realspace_image
            if im_type == "image"
            else self.unscaled_diffraction_image
        )
        with TiffWriter(filename) as tw:
            tw.write(raw)
    else:
        raise RuntimeError(f"Unrecognised export format: {im_format!r}")


# ---------------------------------------------------------------------------
# File dialogs
# ---------------------------------------------------------------------------

def show_file_dialog(self) -> str | None:
    filename, _ = QFileDialog.getOpenFileName(
        self, "Open 4D-STEM Data", "", _OPEN_FILTER
    )
    return filename if filename else None


def get_savefile_name(self, file_format: str) -> str | None:
    filters = {
        "Raw float32":    "RAW File (*.raw *.f32);;Any file (*)",
        "HDF5":           "HDF5 File (*.h5 *.hdf5);;Any file (*)",
        "PNG (display)":  "PNG File (*.png);;Any file (*)",
        "TIFF (display)": "TIFF File (*.tiff *.tif);;Any file (*)",
        "TIFF (raw)":     "TIFF File (*.tiff *.tif);;Any file (*)",
    }
    defaults = {
        "Raw float32":    ".raw",
        "HDF5":           ".h5",
        "PNG (display)":  ".png",
        "TIFF (display)": ".tiff",
        "TIFF (raw)":     ".tiff",
    }
    fname, _ = QFileDialog.getSaveFileName(
        self, "Save file", "", filters.get(file_format, "Any file (*)")
    )
    if not fname:
        return None
    if not os.path.splitext(fname)[1]:
        fname += defaults.get(file_format, "")
    return fname


def show_keyboard_map(self):
    keymap = KeyboardMapMenu(parent=self)
    keymap.open()
