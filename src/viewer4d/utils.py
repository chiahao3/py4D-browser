"""
Shared GUI utilities: detector geometry types, helper widgets.
"""

import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication
from PyQt5.QtCore import pyqtSignal, Qt
from typing import NotRequired, TypedDict
from enum import Enum


class DetectorShape(Enum):
    RECTANGULAR = "rectangular"
    POINT = "point"
    CIRCLE = "circle"
    ANNULUS = "annulus"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.replace("&", "").lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class DetectorMode(Enum):
    INTEGRATING = "integrating"
    MAXIMUM = "maximum"
    CoM = "com"
    CoMx = "comx"
    CoMy = "comy"
    ICOM = "icom"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.replace("&", "").replace(" ", "").lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


RectangleGeometry = TypedDict(
    "RectangleGeometry",
    {"xmin": float, "xmax": float, "ymin": float, "ymax": float},
)
CircleGeometry = TypedDict(
    "CircleGeometry",
    {"x": float, "y": float, "R": float},
)
AnnulusGeometry = TypedDict(
    "AnnulusGeometry",
    {"x": float, "y": float, "R_inner": float, "R_outer": float},
)
PointGeometry = TypedDict(
    "PointGeometry",
    {"x": float, "y": float},
)
DetectorInfo = TypedDict(
    "DetectorInfo",
    {
        "shape": DetectorShape,
        "mode": DetectorMode,
        "geometry": RectangleGeometry | CircleGeometry | AnnulusGeometry | PointGeometry,
        "slice": NotRequired[list],
        "mask": NotRequired[np.ndarray],
        "point": NotRequired[list],
    },
)


class StatusBarWriter:
    """File-like object that writes tqdm progress to the Qt status bar."""

    def __init__(self, statusBar):
        self.statusBar = statusBar
        self.app = QApplication.instance()

    def write(self, message):
        self.statusBar.showMessage(message, 1_000)
        self.app.processEvents()

    def flush(self):
        pass


class VLine(QFrame):
    """Thin vertical divider for the status bar."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(self.VLine | self.Sunken)


class LatchingButton(QPushButton):
    """
    Momentary push-button that latches on with Shift+click.
    Emits `activated` on every trigger.
    """

    activated = pyqtSignal()

    def __init__(self, *args, **kwargs):
        self.status_bar = kwargs.pop("status_bar", None)
        self.latched = kwargs.pop("latched", False)
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.clicked.connect(self.on_click)
        if self.latched:
            self.setChecked(True)
            self.activated.emit()

    def on_click(self, *args):
        modifiers = QApplication.keyboardModifiers()
        if self.latched:
            self.setChecked(False)
            self.latched = False
        else:
            if modifiers == Qt.ShiftModifier:
                self.setChecked(True)
                self.latched = True
                self.activated.emit()
            else:
                self.setChecked(False)
                self.latched = False
                self.activated.emit()
                if self.status_bar is not None:
                    self.status_bar.showMessage("Shift+click to keep on", 5_000)


def pg_point_roi(view_box, center=(-0.5, -0.5), pen=(0, 9), hoverPen=None):
    """Small circular ROI used as a point selector."""
    circ_roi = pg.CircleROI(center, (2, 2), movable=True, pen=pen, hoverPen=hoverPen)
    h = circ_roi.addTranslateHandle((0.5, 0.5))
    h.pen = pg.mkPen("r")
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi


def make_detector(shape: tuple, mode: str, geometry) -> np.ndarray:
    """Build a boolean detector mask from shape/geometry specification."""
    match mode, geometry:
        case ["point", (qx, qy)]:
            mask = np.zeros(shape, dtype=np.bool_)
            mask[qx, qy] = True
        case ["point", geom]:
            raise ValueError(f"Point geometry must be (qx,qy), got {geom}")

        case [("circle" | "circular"), ((qx, qy), r)]:
            ix, iy = np.indices(shape)
            mask = np.hypot(ix - qx, iy - qy) <= r
        case [("circle" | "circular"), geom]:
            raise ValueError(f"Circle geometry must be ((qx,qy),r), got {geom}")

        case [("annulus" | "annular"), ((qx, qy), (ri, ro))]:
            ix, iy = np.indices(shape)
            ir = np.hypot(ix - qx, iy - qy)
            mask = np.logical_and(ir >= ri, ir <= ro)
        case [("annulus" | "annular"), geom]:
            raise ValueError(f"Annulus geometry must be ((qx,qy),(ri,ro)), got {geom}")

        case [("rectangle" | "square" | "rectangular"), (xmin, xmax, ymin, ymax)]:
            mask = np.zeros(shape, dtype=np.bool_)
            mask[xmin:xmax, ymin:ymax] = True
        case [("rectangle" | "square" | "rectangular"), geom]:
            raise ValueError(f"Rectangle geometry must be (xmin,xmax,ymin,ymax), got {geom}")

        case ["mask", mask_arr]:
            mask = mask_arr

        case unknown:
            raise ValueError(f"mode/geometry not understood: {unknown}")

    return mask


def complex_to_Lab(
    im, amin=None, amax=None, gamma=1.0, L_scale=100, ab_scale=64, uniform_L=None
):
    """Convert complex array to Lab colour image (magnitude→L, phase→hue)."""
    from skimage.color import lab2rgb
    from matplotlib.colors import Normalize
    import warnings

    Lab = np.zeros(im.shape + (3,), dtype=np.float64)
    angle = np.angle(im)

    L = Normalize(vmin=amin, vmax=amax, clip=True)(np.abs(im)) ** gamma
    L = Normalize()(L)
    ab_prescale = 0.5

    Lab[..., 0] = uniform_L or L * L_scale
    Lab[..., 1] = np.cos(angle) * ab_scale * ab_prescale
    Lab[..., 2] = np.sin(angle) * ab_scale * ab_prescale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = lab2rgb(Lab)

    return rgb
