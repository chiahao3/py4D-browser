"""
Angle-Corrected Bright-Field (acBF) plugin.

acBF decomposes the BF / annular-BF detector into concentric radial rings.
Each ring is treated as a separate tcBF reconstruction with a shift magnitude
proportional to its scattering semi-angle, then the results are averaged.

This gives better contrast than plain tcBF when:
  - The outer BF disk has reversed contrast (strong diffraction conditions)
  - Using an annular BF detector

Requires calibrated data (Calibrate… plugin) for the automatic mode.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QWidget, QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QGridLayout, QCheckBox, QComboBox,
)

from viewer4d.utils import DetectorShape, DetectorInfo


def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


class acBFPlugin(QWidget):
    plugin_id        = "viewer4d.acBF"
    uses_plugin_menu = True
    display_name     = "Angle-Corrected BF"

    def __init__(self, parent, plugin_menu, **kwargs):
        super().__init__()
        self.parent = parent

        action = plugin_menu.addAction(
            "acBF (GPU)…" if _torch_available() else "acBF (CPU)…"
        )
        action.triggered.connect(self._launch)

    def close(self):
        pass

    def _launch(self):
        parent = self.parent
        detector: DetectorInfo = parent.get_diffraction_detector()

        if detector["shape"] is DetectorShape.POINT:
            parent.statusBar().showMessage("acBF requires an area detector!", 5_000)
            return

        AcBFDialog(parent=parent, mask=detector["mask"]).show()


class AcBFDialog(QDialog):
    def __init__(self, parent, mask: np.ndarray):
        super().__init__(parent=parent)
        self.parent = parent
        self.mask = mask
        self.setWindowTitle("Angle-Corrected BF")

        layout = QVBoxLayout(self)

        info_box = QGroupBox("Parameters")
        layout.addWidget(info_box)
        grid = QGridLayout()
        info_box.setLayout(grid)

        grid.addWidget(QLabel("Beam energy [keV]"), 0, 0, Qt.AlignRight)
        self.energy_box = QLineEdit("300")
        self.energy_box.setValidator(QDoubleValidator(1, 3000, 1))
        grid.addWidget(self.energy_box, 0, 1)

        grid.addWidget(QLabel("Q pixel size [mrad/px]"), 1, 0, Qt.AlignRight)
        cal = parent.datacube.calibration
        q_hint = (
            f"{cal.get_Q_pixel_size():.4g}"
            if cal.get_Q_pixel_units() == "mrad" else ""
        )
        self.q_pix_box = QLineEdit(q_hint)
        self.q_pix_box.setValidator(QDoubleValidator())
        self.q_pix_box.setPlaceholderText("mrad per diffraction pixel")
        grid.addWidget(self.q_pix_box, 1, 1)

        grid.addWidget(QLabel("Rotation [deg]"), 2, 0, Qt.AlignRight)
        self.rotation_box = QLineEdit("0.0")
        self.rotation_box.setValidator(QDoubleValidator())
        grid.addWidget(self.rotation_box, 2, 1)

        grid.addWidget(QLabel("Transpose x/y"), 3, 0, Qt.AlignRight)
        self.transpose_box = QCheckBox()
        grid.addWidget(self.transpose_box, 3, 1)

        grid.addWidget(QLabel("Number of rings"), 4, 0, Qt.AlignRight)
        self.rings_box = QLineEdit("8")
        self.rings_box.setValidator(QIntValidator(1, 64))
        grid.addWidget(self.rings_box, 4, 1)

        grid.addWidget(QLabel("Pad Images"), 5, 0, Qt.AlignRight)
        self.pad_checkbox = QCheckBox()
        grid.addWidget(self.pad_checkbox, 5, 1)

        grid.addWidget(QLabel("Backend"), 6, 0, Qt.AlignRight)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(
            ["PyTorch (GPU/CPU)", "NumPy (CPU)"] if _torch_available() else ["NumPy (CPU)"]
        )
        grid.addWidget(self.backend_combo, 6, 1)

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

        q_txt = self.q_pix_box.text()
        if not q_txt:
            parent.statusBar().showMessage(
                "Please enter Q pixel size in mrad/px.", 5_000
            )
            return

        alpha_mrad_per_px = float(q_txt)
        alpha_rad_per_px  = alpha_mrad_per_px * 1e-3

        energy_eV    = float(self.energy_box.text() or 300) * 1e3
        rotation_rad = np.radians(float(self.rotation_box.text() or 0.0))
        transpose    = bool(self.transpose_box.checkState())
        n_rings      = max(1, int(self.rings_box.text() or 8))
        pad          = bool(self.pad_checkbox.checkState())
        use_torch    = "PyTorch" in self.backend_combo.currentText()

        parent.statusBar().showMessage(
            f"acBF: {n_rings} rings, reconstructing…"
        )
        parent.qtapp.processEvents()

        data = np.asarray(parent.datacube.data)

        if use_torch and _torch_available():
            from viewer4d.processing.acbf import acbf_torch
            recon = acbf_torch(
                data, self.mask, alpha_rad_per_px, energy_eV,
                rotation_rad=rotation_rad, transpose=transpose,
                pad=pad, n_rings=n_rings,
            )
        else:
            from viewer4d.processing.acbf import acbf_numpy
            recon = acbf_numpy(
                data, self.mask, alpha_rad_per_px, energy_eV,
                rotation_rad=rotation_rad, transpose=transpose,
                pad=pad, n_rings=n_rings,
            )

        parent.set_virtual_image(recon, reset=True)
        parent.statusBar().showMessage("acBF: done.", 5_000)
        self.close()
