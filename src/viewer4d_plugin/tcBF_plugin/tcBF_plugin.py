"""
Tilt-Corrected Bright-Field (tcBF) plugin.

Provides two reconstruction modes:
  - Manual: user supplies rotation, transpose, and max_shift
  - Automatic: shifts are derived from calibrated convergence semi-angle
               using the GPU-accelerated batch FFT backend

The reconstruction itself lives in viewer4d.processing.tcbf so it can also
be called programmatically from notebooks or scripts.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QWidget, QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QGridLayout, QCheckBox, QComboBox,
)

from viewer4d.utils import DetectorShape, DetectorInfo
from viewer4d.processing.tcbf import compute_shift_field


def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


class tcBFPlugin(QWidget):
    plugin_id       = "viewer4d.tcBF"
    uses_plugin_menu = True
    display_name    = "Tilt-Corrected BF"

    def __init__(self, parent, plugin_menu, **kwargs):
        super().__init__()
        self.parent = parent

        manual_action = plugin_menu.addAction("Manual tcBF…")
        manual_action.triggered.connect(self._launch_manual)

        auto_action = plugin_menu.addAction("Automatic tcBF (GPU)" if _torch_available() else "Automatic tcBF (CPU)")
        auto_action.triggered.connect(self._launch_auto)

    def close(self):
        pass

    def _launch_manual(self):
        ManualTCBFDialog(parent=self.parent).show()

    def _launch_auto(self):
        parent = self.parent
        detector: DetectorInfo = parent.get_diffraction_detector()

        if detector["shape"] is DetectorShape.POINT:
            parent.statusBar().showMessage("tcBF requires an area detector!", 5_000)
            return

        cal = parent.datacube.calibration
        if (
            cal.get_R_pixel_units() == "pixels"
            or cal.get_Q_pixel_units() == "pixels"
        ):
            parent.statusBar().showMessage(
                "Auto tcBF requires calibrated data (Calibrate… plugin)", 5_000
            )
            return

        AutoTCBFDialog(parent=parent, mask=detector["mask"]).show()


class ManualTCBFDialog(QDialog):
    """Manual tcBF: user supplies rotation angle, transpose flag, and max shift."""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.setWindowTitle("Manual tcBF")

        layout = QVBoxLayout(self)
        params = QGroupBox("Parameters")
        layout.addWidget(params)
        grid = QGridLayout()
        params.setLayout(grid)

        grid.addWidget(QLabel("Rotation [deg]"), 0, 0, Qt.AlignRight)
        self.rotation_box = QLineEdit("0.0")
        self.rotation_box.setValidator(QDoubleValidator())
        grid.addWidget(self.rotation_box, 0, 1)

        grid.addWidget(QLabel("Transpose x/y"), 1, 0, Qt.AlignRight)
        self.transpose_box = QCheckBox()
        grid.addWidget(self.transpose_box, 1, 1)

        grid.addWidget(QLabel("Max Shift [px]"), 2, 0, Qt.AlignRight)
        self.max_shift_box = QLineEdit()
        self.max_shift_box.setValidator(QDoubleValidator())
        grid.addWidget(self.max_shift_box, 2, 1)

        grid.addWidget(QLabel("Pad Images"), 3, 0, Qt.AlignRight)
        self.pad_checkbox = QCheckBox()
        grid.addWidget(self.pad_checkbox, 3, 1)

        grid.addWidget(QLabel("Backend"), 4, 0, Qt.AlignRight)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(
            ["PyTorch (GPU/CPU)", "NumPy (CPU)"] if _torch_available() else ["NumPy (CPU)"]
        )
        grid.addWidget(self.backend_combo, 4, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel = QPushButton("Cancel")
        cancel.pressed.connect(self.close)
        btn_layout.addWidget(cancel)
        run = QPushButton("Reconstruct")
        run.pressed.connect(self._reconstruct)
        btn_layout.addWidget(run)
        layout.addLayout(btn_layout)

    def _reconstruct(self):
        parent = self.parent
        detector: DetectorInfo = parent.get_diffraction_detector()

        if detector["shape"] is DetectorShape.POINT:
            parent.statusBar().showMessage("tcBF needs an area detector!", 5_000)
            return

        if not self.max_shift_box.text():
            parent.statusBar().showMessage("Please enter Max Shift.", 5_000)
            return

        mask      = detector["mask"]
        rotation  = np.radians(float(self.rotation_box.text() or 0.0))
        transpose = bool(self.transpose_box.checkState())
        max_shift = float(self.max_shift_box.text())
        pad       = bool(self.pad_checkbox.checkState())
        use_torch = "PyTorch" in self.backend_combo.currentText()

        shifts_x, shifts_y = compute_shift_field(
            mask, max_shift, rotation_rad=rotation, transpose=transpose
        )

        parent.statusBar().showMessage("tcBF: reconstructing…")
        parent.qtapp.processEvents()

        data = np.asarray(parent.datacube.data)

        if use_torch and _torch_available():
            from viewer4d.processing.tcbf import tcbf_torch
            recon = tcbf_torch(data, mask, shifts_x, shifts_y, pad=pad)
        else:
            from viewer4d.processing.tcbf import tcbf_numpy
            recon = tcbf_numpy(data, mask, shifts_x, shifts_y, pad=pad)

        parent.set_virtual_image(recon, reset=True)
        parent.statusBar().showMessage("tcBF: done.", 5_000)
        self.close()


class AutoTCBFDialog(QDialog):
    """
    Automatic tcBF: infers max shift from calibration (convergence semi-angle).
    Requires calibrated data.
    """

    def __init__(self, parent, mask: np.ndarray):
        super().__init__(parent=parent)
        self.parent = parent
        self.mask = mask
        self.setWindowTitle("Automatic tcBF")

        cal = parent.datacube.calibration
        layout = QVBoxLayout(self)

        params = QGroupBox("Parameters (inferred from calibration)")
        layout.addWidget(params)
        grid = QGridLayout()
        params.setLayout(grid)

        grid.addWidget(QLabel("Rotation [deg]"), 0, 0, Qt.AlignRight)
        self.rotation_box = QLineEdit("0.0")
        self.rotation_box.setValidator(QDoubleValidator())
        grid.addWidget(self.rotation_box, 0, 1)

        grid.addWidget(QLabel("Transpose x/y"), 1, 0, Qt.AlignRight)
        self.transpose_box = QCheckBox()
        grid.addWidget(self.transpose_box, 1, 1)

        grid.addWidget(QLabel("Pad Images"), 2, 0, Qt.AlignRight)
        self.pad_checkbox = QCheckBox()
        grid.addWidget(self.pad_checkbox, 2, 1)

        grid.addWidget(QLabel("Backend"), 3, 0, Qt.AlignRight)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(
            ["PyTorch (GPU/CPU)", "NumPy (CPU)"] if _torch_available() else ["NumPy (CPU)"]
        )
        grid.addWidget(self.backend_combo, 3, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel = QPushButton("Cancel")
        cancel.pressed.connect(self.close)
        btn_layout.addWidget(cancel)
        run = QPushButton("Reconstruct")
        run.pressed.connect(self._reconstruct)
        btn_layout.addWidget(run)
        layout.addLayout(btn_layout)

    def _reconstruct(self):
        parent = self.parent
        cal = parent.datacube.calibration

        # Estimate max_shift from R_px/Q_px calibrations:
        # max_shift [R_px] ≈ (Q_px_size / R_px_size) * mask_radius_px
        # This is approximate; users can refine with Manual mode.
        mask   = self.mask
        q_size = cal.get_Q_pixel_size()
        r_size = cal.get_R_pixel_size()
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            parent.statusBar().showMessage("Mask is empty!", 5_000)
            return

        cx = mask.shape[0] // 2
        cy = mask.shape[1] // 2
        r_max = np.max(np.hypot(mask_coords[:, 0] - cx, mask_coords[:, 1] - cy))
        max_shift = r_max * (q_size / r_size) if r_size > 0 else r_max

        rotation  = np.radians(float(self.rotation_box.text() or 0.0))
        transpose = bool(self.transpose_box.checkState())
        pad       = bool(self.pad_checkbox.checkState())
        use_torch = "PyTorch" in self.backend_combo.currentText()

        shifts_x, shifts_y = compute_shift_field(
            mask, max_shift, rotation_rad=rotation, transpose=transpose
        )

        parent.statusBar().showMessage(
            f"Auto tcBF: max_shift={max_shift:.1f} px, reconstructing…"
        )
        parent.qtapp.processEvents()

        data = np.asarray(parent.datacube.data)

        if use_torch and _torch_available():
            from viewer4d.processing.tcbf import tcbf_torch
            recon = tcbf_torch(data, mask, shifts_x, shifts_y, pad=pad)
        else:
            from viewer4d.processing.tcbf import tcbf_numpy
            recon = tcbf_numpy(data, mask, shifts_x, shifts_y, pad=pad)

        parent.set_virtual_image(recon, reset=True)
        parent.statusBar().showMessage("Auto tcBF: done.", 5_000)
        self.close()
