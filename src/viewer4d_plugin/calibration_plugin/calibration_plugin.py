"""
Calibration plugin — set real-space and reciprocal-space pixel sizes.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QWidget, QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QGroupBox, QGridLayout,
)

from viewer4d.utils import DetectorShape, DetectorInfo, CircleGeometry


class CalibrationPlugin(QWidget):
    plugin_id         = "viewer4d.calibration"
    uses_single_action = True
    display_name      = "Calibrate..."

    def __init__(self, parent, plugin_action, **kwargs):
        super().__init__()
        self.parent = parent
        plugin_action.triggered.connect(self._launch)

    def close(self):
        pass

    def _launch(self):
        detector: DetectorInfo = self.parent.get_diffraction_detector()
        selector_size = None
        if detector["shape"] is DetectorShape.CIRCLE:
            g: CircleGeometry = detector["geometry"]
            selector_size = g["R"]
        else:
            self.parent.statusBar().showMessage(
                "Use a Circle selection to calibrate from a known spacing...", 5_000
            )
        dialog = CalibrateDialog(
            self.parent.datacube, parent=self.parent,
            diffraction_selector_size=selector_size,
        )
        dialog.open()


class CalibrateDialog(QDialog):
    def __init__(self, datacube, parent, diffraction_selector_size=None):
        super().__init__(parent=parent)
        self.datacube = datacube
        self.parent = parent
        self.diffraction_selector_size = diffraction_selector_size
        self.setWindowTitle("Calibrate")

        layout = QVBoxLayout(self)

        # ---- Real space ----
        rs_box = QGroupBox("Real Space")
        layout.addWidget(rs_box)
        rs_layout = QHBoxLayout()
        rs_box.setLayout(rs_layout)
        rs_grid = QGridLayout()
        rs_layout.addLayout(rs_grid)

        rs_grid.addWidget(QLabel("Pixel Size"), 0, 0, Qt.AlignRight)
        self.rs_pix = QLineEdit()
        self.rs_pix.setValidator(QDoubleValidator())
        rs_grid.addWidget(self.rs_pix, 0, 1)

        rs_grid.addWidget(QLabel("Full Width"), 1, 0, Qt.AlignRight)
        self.rs_fov = QLineEdit()
        rs_grid.addWidget(self.rs_fov, 1, 1)

        self.rs_unit = QComboBox()
        self.rs_unit.addItems(["Å", "nm"])
        self.rs_unit.setMinimumContentsLength(5)
        rs_layout.addWidget(self.rs_unit)

        # ---- Diffraction ----
        diff_box = QGroupBox("Diffraction")
        layout.addWidget(diff_box)
        diff_layout = QHBoxLayout()
        diff_box.setLayout(diff_layout)
        diff_grid = QGridLayout()
        diff_layout.addLayout(diff_grid)

        diff_grid.addWidget(QLabel("Pixel Size"), 0, 0, Qt.AlignRight)
        self.diff_pix = QLineEdit()
        self.diff_pix.setValidator(QDoubleValidator())
        diff_grid.addWidget(self.diff_pix, 0, 1)

        diff_grid.addWidget(QLabel("Full Width"), 1, 0, Qt.AlignRight)
        self.diff_fov = QLineEdit()
        diff_grid.addWidget(self.diff_fov, 1, 1)

        diff_grid.addWidget(QLabel("Selection Radius"), 2, 0, Qt.AlignRight)
        self.diff_sel = QLineEdit()
        self.diff_sel.setEnabled(diffraction_selector_size is not None)
        diff_grid.addWidget(self.diff_sel, 2, 1)

        self.diff_unit = QComboBox()
        self.diff_unit.addItems(["mrad", "Å⁻¹"])
        self.diff_unit.setMinimumContentsLength(5)
        diff_layout.addWidget(self.diff_unit)

        # ---- Buttons ----
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel = QPushButton("Cancel")
        cancel.pressed.connect(self.close)
        btn_layout.addWidget(cancel)
        done = QPushButton("Done")
        done.pressed.connect(self._apply)
        btn_layout.addWidget(done)
        layout.addLayout(btn_layout)

        # Connections
        self.rs_pix.textEdited.connect(self._rs_pix_changed)
        self.rs_fov.textEdited.connect(self._rs_fov_changed)
        self.diff_pix.textEdited.connect(self._diff_pix_changed)
        self.diff_fov.textEdited.connect(self._diff_fov_changed)
        self.diff_sel.textEdited.connect(self._diff_sel_changed)

    # ---- Field callbacks ----

    def _rs_pix_changed(self, v):
        if not v:
            return
        pix = float(v)
        self.rs_fov.setText(f"{pix * self.datacube.R_Ny:g}")

    def _rs_fov_changed(self, v):
        if not v:
            return
        self.rs_pix.setText(f"{float(v) / self.datacube.R_Ny:g}")

    def _diff_pix_changed(self, v):
        if not v:
            return
        pix = float(v)
        self.diff_fov.setText(f"{pix * self.datacube.Q_Ny:g}")
        if self.diffraction_selector_size:
            self.diff_sel.setText(f"{pix * self.diffraction_selector_size:g}")

    def _diff_fov_changed(self, v):
        if not v:
            return
        pix = float(v) / self.datacube.Q_Ny
        self.diff_pix.setText(f"{pix:g}")
        if self.diffraction_selector_size:
            self.diff_sel.setText(f"{pix * self.diffraction_selector_size:g}")

    def _diff_sel_changed(self, v):
        if not v or not self.diffraction_selector_size:
            return
        pix = float(v) / self.diffraction_selector_size
        self.diff_pix.setText(f"{pix:g}")
        self.diff_fov.setText(f"{pix * self.datacube.Q_Nx:g}")

    # ---- Apply ----

    def _apply(self):
        cal = self.datacube.calibration

        rs_txt = self.rs_pix.text()
        if rs_txt:
            cal.set_R_pixel_size(float(rs_txt))
            cal.set_R_pixel_units(
                self.rs_unit.currentText().replace("Å", "A")
            )

        diff_txt = self.diff_pix.text()
        if diff_txt:
            cal.set_Q_pixel_size(float(diff_txt))
            unit_map = {"mrad": "mrad", "Å⁻¹": "A^-1"}
            cal.set_Q_pixel_units(unit_map[self.diff_unit.currentText()])

        self.parent.update_scalebars()
        self.close()
