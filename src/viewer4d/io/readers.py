"""
File readers for 4D-STEM data.

Supported formats (via rosettasciio):
  .dm3 / .dm4   — Gatan DigitalMicrograph
  .mib          — Merlin Image Binary (Medipix)
  .emd          — Electron Microscopy Dataset (Velox / Berkeley EMD)
  .h5 / .hdf5   — Generic HDF5 (scanned for 4D datasets)
  .npy / .npz   — NumPy arrays
  .raw          — Raw float32 binary (shape must be set manually)

The primary entry point is ``load_file()``.
"""

from __future__ import annotations

import os
import numpy as np
import h5py

from viewer4d.datacube import DataCube, Calibration


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(filepath: str, mmap: bool = False, binning: int = 1) -> DataCube:
    """
    Load a 4D-STEM dataset from *filepath*.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    mmap : bool
        If True, memory-map the file rather than reading into RAM.
        Only supported for HDF5 and NPY formats.
    binning : int
        Spatial binning factor applied to the diffraction dimension on load.
        Only supported for formats read via rosettasciio.

    Returns
    -------
    DataCube
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext in (".h5", ".hdf5", ".emd"):
        return _load_hdf5(filepath, mmap=mmap)

    if ext == ".npy":
        return DataCube(np.load(filepath, mmap_mode="r" if mmap else None))

    if ext == ".npz":
        archive = np.load(filepath)
        # Use the first array that is 4-D
        for key, arr in archive.items():
            if arr.ndim == 4:
                return DataCube(arr)
        raise ValueError(f"No 4D array found in {filepath}")

    # All other formats go through rosettasciio
    return _load_via_rosettasciio(filepath, mmap=mmap, binning=binning)


# ---------------------------------------------------------------------------
# HDF5 / EMD
# ---------------------------------------------------------------------------

def _load_hdf5(filepath: str, mmap: bool = False) -> DataCube:
    """Load the first 4D dataset found inside an HDF5 file."""
    f = h5py.File(filepath, "r")
    datasets = _find_nd_datasets(f, N=4)

    if datasets:
        dset = datasets[0]
        print(f"  Reading 4D dataset at: {dset.name}")
        data = dset if mmap else dset[()]
        cal = _calibration_from_hdf5(dset)
        return DataCube(data, calibration=cal)

    # Fall back to 3D — caller (GUI) will prompt for reshape
    datasets_3d = _find_nd_datasets(f, N=3)
    if datasets_3d:
        print(f"  No 4D dataset found; returning 3D dataset (needs reshape).")
        data = datasets_3d[0] if mmap else datasets_3d[0][()]
        return DataCube(data.reshape(1, *data.shape))

    raise ValueError(f"No 4D (or 3D) dataset found in {filepath}")


def _find_nd_datasets(node, N: int = 4, results: list | None = None) -> list:
    """Recursively collect all h5py Datasets with N dimensions."""
    if results is None:
        results = []
    for key in node.keys():
        item = node[key]
        if isinstance(item, h5py.Dataset) and len(item.shape) == N:
            results.append(item)
        elif isinstance(item, h5py.Group):
            _find_nd_datasets(item, N=N, results=results)
    return results


def _calibration_from_hdf5(dset: h5py.Dataset) -> Calibration:
    """Try to extract pixel calibrations from HDF5 metadata."""
    cal = Calibration()

    # --- py4DSTEM / Berkeley EMD format ---
    try:
        if "emd_group_type" in dset.parent.attrs:
            meta = dset.parent.parent["metadatabundle"]["calibration"]
            cal.set_R_pixel_size(float(meta["R_pixel_size"][()]))
            cal.set_R_pixel_units(meta["R_pixel_units"][()].decode())
            cal.set_Q_pixel_size(float(meta["Q_pixel_size"][()]))
            cal.set_Q_pixel_units(meta["Q_pixel_units"][()].decode())
            return cal
    except Exception:
        pass

    # --- abTEM / generic "sampling" convention ---
    try:
        if "sampling" in dset.parent and "units" in dset.parent:
            sampling = dset.parent["sampling"]
            units = dset.parent["units"]
            cal.set_R_pixel_size(float(sampling[0]))
            cal.set_R_pixel_units(units[0].decode().replace("Å", "A"))
            cal.set_Q_pixel_size(float(sampling[3]))
            cal.set_Q_pixel_units(units[3].decode())
            return cal
    except Exception:
        pass

    return cal


# ---------------------------------------------------------------------------
# rosettasciio (dm3/dm4/mib and everything else)
# ---------------------------------------------------------------------------

def _load_via_rosettasciio(
    filepath: str, mmap: bool = False, binning: int = 1
) -> DataCube:
    """
    Load via rosettasciio.  Falls back to a helpful error if the package is
    missing or the format is unsupported.
    """
    try:
        import rosettasciio as rsciio
    except ImportError:
        raise ImportError(
            "rosettasciio is required to load this file format. "
            "Install it with:  pip install rosettasciio[all]"
        )

    ext = os.path.splitext(filepath)[-1].lower()

    # Use format-specific readers when possible for cleaner error messages
    reader_map = {
        ".dm3": "gatan",
        ".dm4": "gatan",
        ".mib": "mib",
    }
    fmt = reader_map.get(ext)

    try:
        if fmt:
            reader_mod = _rsciio_reader(rsciio, fmt)
            signals = reader_mod.file_reader(filepath)
        else:
            # Generic rsciio.load (requires hyperspy-style dispatcher)
            signals = _rsciio_generic_load(rsciio, filepath)
    except Exception as exc:
        raise RuntimeError(
            f"rosettasciio failed to read {filepath!r}: {exc}"
        ) from exc

    if not signals:
        raise ValueError(f"rosettasciio returned no data for {filepath!r}")

    # Find first 4D signal
    sig = _pick_4d_signal(signals)
    data = np.asarray(sig["data"])

    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D data from {filepath!r}, got shape {data.shape}"
        )

    if binning > 1:
        data = _bin_diffraction(data, binning)

    cal = _calibration_from_rsciio_axes(sig.get("axes", []))
    return DataCube(data, calibration=cal)


def _rsciio_reader(rsciio, fmt: str):
    """Import a rosettasciio format submodule."""
    import importlib
    return importlib.import_module(f"rosettasciio.{fmt}")


def _rsciio_generic_load(rsciio, filepath: str) -> list:
    """
    Try the hyperspy-style dispatcher inside rosettasciio.
    Older versions don't expose a top-level load(); newer ones do.
    """
    # Try rsciio's own dispatcher
    for attr in ("load", "io_plugins"):
        if hasattr(rsciio, attr):
            break
    # Delegate to hyperspy if present
    try:
        import hyperspy.io_plugins as _hp_io
        from hyperspy.misc.io.tools import load_with_reader
        return load_with_reader(filepath)
    except ImportError:
        pass
    raise RuntimeError(
        "Cannot auto-detect file format. Install hyperspy or use a "
        "format with a direct rsciio reader (dm3/dm4/mib)."
    )


def _pick_4d_signal(signals: list) -> dict:
    """Return the first signal dict whose data is 4-dimensional."""
    for sig in signals:
        arr = np.asarray(sig["data"])
        if arr.ndim == 4:
            return sig
    # No pure-4D signal; take largest and hope for the best
    signals_sorted = sorted(signals, key=lambda s: np.asarray(s["data"]).size, reverse=True)
    return signals_sorted[0]


def _calibration_from_rsciio_axes(axes: list) -> Calibration:
    """
    Extract R / Q calibration from rosettasciio axes list.

    Convention: axes[0], axes[1] are scan (navigate=True),
                axes[2], axes[3] are diffraction (navigate=False).
    """
    cal = Calibration()
    if len(axes) < 4:
        return cal

    try:
        cal.set_R_pixel_size(float(axes[0].get("scale", 1.0)))
        r_units = str(axes[0].get("units", "pixels") or "pixels")
        cal.set_R_pixel_units(r_units.replace("Å", "A").replace("µm", "um"))
    except Exception:
        pass

    try:
        cal.set_Q_pixel_size(float(axes[2].get("scale", 1.0)))
        q_units = str(axes[2].get("units", "pixels") or "pixels")
        cal.set_Q_pixel_units(q_units.replace("Å", "A").replace("µm", "um"))
    except Exception:
        pass

    return cal


def _bin_diffraction(data: np.ndarray, factor: int) -> np.ndarray:
    """Bin the diffraction (last two) dimensions by *factor*."""
    R_Nx, R_Ny, Q_Nx, Q_Ny = data.shape
    Q_Nx_new = Q_Nx // factor
    Q_Ny_new = Q_Ny // factor
    data = data[:, :, : Q_Nx_new * factor, : Q_Ny_new * factor]
    data = data.reshape(R_Nx, R_Ny, Q_Nx_new, factor, Q_Ny_new, factor)
    return data.mean(axis=(3, 5))
