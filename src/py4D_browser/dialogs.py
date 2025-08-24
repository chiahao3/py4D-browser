from py4DSTEM import DataCube, data
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QCheckBox,
)
from py4D_browser.utils import make_detector, StatusBarWriter


class ResizeDialog(QDialog):
    def __init__(self, size, parent=None):
        super().__init__(parent=parent)

        self.new_size = size
        Nmax = size[0] * size[1]

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Dataset size unknown. Please enter the shape:"))

        box_layout = QHBoxLayout()
        box_layout.addWidget(QLabel("X:"))

        xbox = QSpinBox()
        xbox.setRange(1, Nmax)
        xbox.setSingleStep(1)
        xbox.setKeyboardTracking(False)
        xbox.valueChanged.connect(self.x_box_changed)
        box_layout.addWidget(xbox)

        box_layout.addStretch()
        box_layout.addWidget(QLabel("Y:"))

        ybox = QSpinBox()
        ybox.setRange(1, Nmax)
        ybox.setSingleStep(1)
        ybox.setValue(Nmax)
        ybox.setKeyboardTracking(False)
        ybox.valueChanged.connect(self.y_box_changed)
        box_layout.addWidget(ybox)

        layout.addLayout(box_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        done_button = QPushButton("Done")
        done_button.pressed.connect(self.close)
        button_layout.addWidget(done_button)
        layout.addLayout(button_layout)

        self.x_box = xbox
        self.y_box = ybox
        self.x_box_last = xbox.value()
        self.y_box_last = ybox.value()
        self.N = Nmax

        self.resize(600, 400)

    @classmethod
    def get_new_size(cls, size, parent=None):
        dialog = cls(size=size, parent=parent)
        dialog.exec_()
        return dialog.new_size

    def x_box_changed(self, new_value):
        if new_value == self.x_box_last:
            return
        x_new, y_new = self.get_next_rect(
            new_value, "down" if new_value < self.x_box_last else "up"
        )

        self.x_box_last = x_new
        self.y_box_last = y_new

        self.x_box.setValue(x_new)
        self.y_box.setValue(y_new)

        self.new_size = [x_new, y_new]

    def y_box_changed(self, new_value):
        if new_value == self.y_box_last:
            return
        y_new, x_new = self.get_next_rect(
            new_value, "down" if new_value < self.y_box_last else "up"
        )

        self.x_box_last = x_new
        self.y_box_last = y_new

        self.x_box.setValue(x_new)
        self.y_box.setValue(y_new)

        self.new_size = [x_new, y_new]

    def get_next_rect(self, current, direction):
        # get the next perfect rectangle
        iterator = (
            range(current, 0, -1) if direction == "down" else range(current, self.N + 1)
        )

        for i in iterator:
            if self.N % i == 0:
                return i, self.N // i

        raise ValueError("Factor finding failed, frustratingly.")

class DiffractionFlipsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)

        self.setWindowTitle("Set Diffraction Flips")
        self.resize(300, 200)

        layout = QVBoxLayout(self)

        # Flipud input
        flipud_layout = QHBoxLayout()
        flipud_layout.addWidget(QLabel("Flip Up-Down (0 or 1):"))
        self.flipud_spinbox = QSpinBox()
        self.flipud_spinbox.setRange(0, 1)
        self.flipud_spinbox.setValue(current_settings.get("flipud", 0))  # Set current value
        flipud_layout.addWidget(self.flipud_spinbox)
        layout.addLayout(flipud_layout)

        # Fliplr input
        fliplr_layout = QHBoxLayout()
        fliplr_layout.addWidget(QLabel("Flip Left-Right (0 or 1):"))
        self.fliplr_spinbox = QSpinBox()
        self.fliplr_spinbox.setRange(0, 1)
        self.fliplr_spinbox.setValue(current_settings.get("fliplr", 0))  # Set current value
        fliplr_layout.addWidget(self.fliplr_spinbox)
        layout.addLayout(fliplr_layout)

        # Transpose input
        transpose_layout = QHBoxLayout()
        transpose_layout.addWidget(QLabel("Transpose (0 or 1):"))
        self.transpose_spinbox = QSpinBox()
        self.transpose_spinbox.setRange(0, 1)
        self.transpose_spinbox.setValue(current_settings.get("transpose", 0))  # Set current value
        transpose_layout.addWidget(self.transpose_spinbox)
        layout.addLayout(transpose_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def get_values(self):
        return {
            "flipud": self.flipud_spinbox.value(),
            "fliplr": self.fliplr_spinbox.value(),
            "transpose": self.transpose_spinbox.value(),
        }