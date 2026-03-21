"""
Dialogs used by the main GUI: ResizeDialog and BinDialog.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QSpinBox, QLabel, QPushButton,
)


class ResizeDialog(QDialog):
    """Prompt the user to specify a 2D scan shape for a 3D or unknown dataset."""

    def __init__(self, size: list[int], parent=None):
        super().__init__(parent=parent)
        self.new_size = list(size)
        Nmax = size[0] * size[1]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Dataset size unknown. Please enter the scan shape:"))

        box_layout = QHBoxLayout()
        box_layout.addWidget(QLabel("X:"))

        xbox = QSpinBox()
        xbox.setRange(1, Nmax)
        xbox.setSingleStep(1)
        xbox.setKeyboardTracking(False)
        xbox.valueChanged.connect(self._x_changed)
        box_layout.addWidget(xbox)

        box_layout.addStretch()
        box_layout.addWidget(QLabel("Y:"))

        ybox = QSpinBox()
        ybox.setRange(1, Nmax)
        ybox.setSingleStep(1)
        ybox.setValue(Nmax)
        ybox.setKeyboardTracking(False)
        ybox.valueChanged.connect(self._y_changed)
        box_layout.addWidget(ybox)

        layout.addLayout(box_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        done = QPushButton("Done")
        done.pressed.connect(self.close)
        btn_layout.addWidget(done)
        layout.addLayout(btn_layout)

        self._xbox = xbox
        self._ybox = ybox
        self._x_last = xbox.value()
        self._y_last = ybox.value()
        self._N = Nmax
        self.resize(480, 200)

    @classmethod
    def get_new_size(cls, size: list[int], parent=None) -> list[int]:
        dialog = cls(size=size, parent=parent)
        dialog.exec_()
        return dialog.new_size

    def _x_changed(self, val):
        if val == self._x_last:
            return
        x, y = self._next_rect(val, "down" if val < self._x_last else "up")
        self._x_last, self._y_last = x, y
        self._xbox.setValue(x)
        self._ybox.setValue(y)
        self.new_size = [x, y]

    def _y_changed(self, val):
        if val == self._y_last:
            return
        y, x = self._next_rect(val, "down" if val < self._y_last else "up")
        self._x_last, self._y_last = x, y
        self._xbox.setValue(x)
        self._ybox.setValue(y)
        self.new_size = [x, y]

    def _next_rect(self, current: int, direction: str):
        iterator = (
            range(current, 0, -1) if direction == "down" else range(current, self._N + 1)
        )
        for i in iterator:
            if self._N % i == 0:
                return i, self._N // i
        raise ValueError("Factor search failed.")
