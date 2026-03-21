"""
View update logic for the main DataViewer window.

All heavy computation (CoM, iCOM) is done with plain numpy for interactive
speed.  GPU-accelerated routines live in viewer4d.processing and are invoked
from the tcBF / acBF plugins instead.
"""

from __future__ import annotations

import os
from itertools import product

import pyqtgraph as pg
import numpy as np
from functools import partial
from PyQt5.QtWidgets import QApplication, QToolTip
from PyQt5 import QtCore
from PyQt5.QtGui import QCursor
from tqdm import tqdm

from viewer4d.utils import (
    pg_point_roi,
    make_detector,
    complex_to_Lab,
    StatusBarWriter,
    DetectorShape,
    DetectorMode,
    DetectorInfo,
    RectangleGeometry,
    CircleGeometry,
    AnnulusGeometry,
    PointGeometry,
)
from viewer4d.processing.virtual_imaging import icom_from_com


# ---------------------------------------------------------------------------
# Detector geometry readers
# ---------------------------------------------------------------------------

def get_diffraction_detector(self) -> DetectorInfo:
    """Return the current diffraction-space detector configuration."""
    shape = DetectorShape(self.detector_shape_group.checkedAction().text())
    mode = DetectorMode(self.detector_mode_group.checkedAction().text())

    match shape:
        case DetectorShape.POINT:
            roi_state = self.virtual_detector_point.saveState()
            y0, x0 = roi_state["pos"]
            xc = int(np.clip(int(x0 + 1), 0, self.datacube.Q_Nx - 1))
            yc = int(np.clip(int(y0 + 1), 0, self.datacube.Q_Ny - 1))
            return DetectorInfo(
                shape=shape, mode=mode,
                point=[xc, yc],
                geometry=PointGeometry(x=xc, y=yc),
            )

        case DetectorShape.RECTANGULAR:
            slices, _ = self.virtual_detector_roi.getArraySlice(
                self.datacube.data[0, 0, :, :].T,
                self.diffraction_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices
            mask = np.zeros(self.datacube.Qshape, dtype=np.bool_)
            mask[slice_x, slice_y] = True
            return DetectorInfo(
                shape=shape, mode=mode,
                slice=[slice_x, slice_y],
                mask=mask,
                geometry=RectangleGeometry(
                    xmin=slice_x.start, xmax=slice_x.stop,
                    ymin=slice_y.start, ymax=slice_y.stop,
                ),
            )

        case DetectorShape.CIRCLE:
            R = self.virtual_detector_roi.size()[0] / 2.0
            x0 = self.virtual_detector_roi.pos()[1] + R
            y0 = self.virtual_detector_roi.pos()[0] + R
            mask = make_detector(
                (self.datacube.Q_Nx, self.datacube.Q_Ny), "circle", ((x0, y0), R)
            )
            return DetectorInfo(
                shape=shape, mode=mode,
                mask=mask,
                geometry=CircleGeometry(x=x0, y=y0, R=R),
            )

        case DetectorShape.ANNULUS:
            inner_pos  = self.virtual_detector_roi_inner.pos()
            inner_size = self.virtual_detector_roi_inner.size()
            R_inner = inner_size[0] / 2.0
            x0 = inner_pos[1] + R_inner
            y0 = inner_pos[0] + R_inner
            R_outer = self.virtual_detector_roi_outer.size()[0] / 2.0
            if R_inner <= R_outer:
                R_inner -= 1
            mask = make_detector(
                (self.datacube.Q_Nx, self.datacube.Q_Ny),
                "annulus", ((x0, y0), (R_inner, R_outer)),
            )
            return DetectorInfo(
                shape=shape, mode=mode,
                mask=mask,
                geometry=AnnulusGeometry(x=x0, y=y0, R_inner=R_inner, R_outer=R_outer),
            )

        case _:
            raise ValueError("Detector shape not recognised.")


def get_virtual_image_detector(self) -> DetectorInfo:
    """Return the current real-space (virtual image) selector configuration."""
    shape = DetectorShape(self.rs_detector_shape_group.checkedAction().text())
    mode = DetectorMode(self.realspace_detector_mode_group.checkedAction().text())

    match shape:
        case DetectorShape.POINT:
            roi_state = self.real_space_point_selector.saveState()
            y0, x0 = roi_state["pos"]
            xc = int(np.clip(int(x0 + 1), 0, self.datacube.R_Nx - 1))
            yc = int(np.clip(int(y0 + 1), 0, self.datacube.R_Ny - 1))
            return DetectorInfo(
                shape=shape, mode=mode,
                point=[xc, yc],
                geometry=PointGeometry(x=xc, y=yc),
            )

        case DetectorShape.RECTANGULAR:
            slices, _ = self.real_space_rect_selector.getArraySlice(
                np.zeros(self.datacube.Rshape).T,
                self.real_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices
            mask = np.zeros(self.datacube.Rshape, dtype=np.bool_)
            mask[slice_x, slice_y] = True
            return DetectorInfo(
                shape=shape, mode=mode,
                slice=[slice_x, slice_y],
                mask=mask,
                geometry=RectangleGeometry(
                    xmin=slice_x.start, xmax=slice_x.stop,
                    ymin=slice_y.start, ymax=slice_y.stop,
                ),
            )

        case _:
            raise ValueError("Detector shape not recognised.")


# ---------------------------------------------------------------------------
# Real-space (virtual image) update
# ---------------------------------------------------------------------------

def update_real_space_view(self, reset=False):
    if self.datacube is None:
        return

    detector = self.get_diffraction_detector()

    # CoM modes require linear scaling
    scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
    if (
        detector["mode"] in (DetectorMode.CoM, DetectorMode.CoMx, DetectorMode.CoMy)
        and scaling_mode != "Linear"
    ):
        self.statusBar().showMessage("Warning: setting linear scaling for CoM image")
        self.vimg_scale_linear_action.setChecked(True)

    vimg = None
    match detector["shape"]:
        case DetectorShape.RECTANGULAR:
            slice_x, slice_y = detector["slice"]
            self.diffraction_space_view_text.setText(
                f"Diffraction Slice: [{slice_x.start}:{slice_x.stop},{slice_y.start}:{slice_y.stop}]"
            )
            if detector["mode"] is DetectorMode.INTEGRATING:
                vimg = np.sum(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))
            elif detector["mode"] is DetectorMode.MAXIMUM:
                vimg = np.max(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))

        case DetectorShape.CIRCLE:
            g: CircleGeometry = detector["geometry"]
            self.diffraction_space_view_text.setText(
                f"Diffraction Circle: Centre ({g['x']:.0f},{g['y']:.0f}), R={g['R']:.0f}"
            )

        case DetectorShape.ANNULUS:
            g: AnnulusGeometry = detector["geometry"]
            self.diffraction_space_view_text.setText(
                f"Diffraction Annulus: Centre ({g['x']:.0f},{g['y']:.0f}), "
                f"R=({g['R_inner']:.0f},{g['R_outer']:.0f})"
            )

        case DetectorShape.POINT:
            xc, yc = detector["point"]
            vimg = self.datacube.data[:, :, xc, yc]
            self.diffraction_space_view_text.setText(f"Diffraction: Point [{xc},{yc}]")

        case _:
            raise ValueError("Unknown detector shape.")

    # Mask-based computation (circle, annulus, or CoM modes)
    if vimg is None:
        mask = detector["mask"].astype(np.float32)

        if "MASK_DEBUG" in os.environ:
            self.set_diffraction_image(mask, reset=reset)
            return

        R_Nx, R_Ny = self.datacube.R_Nx, self.datacube.R_Ny
        vimg = np.zeros((R_Nx, R_Ny), dtype=np.float32)

        progress = tqdm(
            product(range(R_Nx), range(R_Ny)),
            total=R_Nx * R_Ny,
            file=StatusBarWriter(self.statusBar()),
            mininterval=0.1,
        )

        if detector["mode"] is DetectorMode.INTEGRATING:
            for rx, ry in progress:
                vimg[rx, ry] = np.sum(self.datacube.data[rx, ry] * mask)

        elif detector["mode"] is DetectorMode.MAXIMUM:
            for rx, ry in progress:
                vimg[rx, ry] = np.max(self.datacube.data[rx, ry] * mask)

        elif detector["mode"] in (
            DetectorMode.CoM, DetectorMode.CoMx, DetectorMode.CoMy, DetectorMode.ICOM
        ):
            ry_coord, rx_coord = np.meshgrid(
                np.arange(self.datacube.Q_Ny), np.arange(self.datacube.Q_Nx)
            )
            CoMx = np.zeros((R_Nx, R_Ny), dtype=np.float32)
            CoMy = np.zeros((R_Nx, R_Ny), dtype=np.float32)
            for rx, ry in progress:
                ar = self.datacube.data[rx, ry] * mask
                tot = ar.sum()
                if tot > 0:
                    CoMx[rx, ry] = np.sum(rx_coord * ar) / tot
                    CoMy[rx, ry] = np.sum(ry_coord * ar) / tot
            CoMx -= CoMx.mean()
            CoMy -= CoMy.mean()

            if detector["mode"] is DetectorMode.CoM:
                vimg = CoMx + 1.0j * CoMy
            elif detector["mode"] is DetectorMode.CoMx:
                vimg = CoMx
            elif detector["mode"] is DetectorMode.CoMy:
                vimg = CoMy
            elif detector["mode"] is DetectorMode.ICOM:
                vimg = icom_from_com(CoMx, CoMy)
        else:
            raise ValueError(f"Detector mode not handled: {detector['mode']}")

    self.set_virtual_image(vimg, reset=reset)


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def set_virtual_image(self, vimg, reset=False):
    self.unscaled_realspace_image = vimg
    self._render_virtual_image(reset=reset)


def _render_virtual_image(self, reset=False):
    vimg = self.unscaled_realspace_image

    if np.isrealobj(vimg):
        scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
        if scaling_mode == "Linear":
            new_view = vimg.copy()
        elif scaling_mode == "Log":
            new_view = np.log2(np.maximum(vimg, self.LOG_SCALE_MIN_VALUE))
        elif scaling_mode == "Square Root":
            new_view = np.sqrt(np.maximum(vimg, 0))
        else:
            raise ValueError(f"Scaling mode not recognised: {scaling_mode!r}")

        auto_level = reset or self.realspace_rescale_button.latched
        self.real_space_widget.setImage(
            new_view.T,
            autoLevels=False,
            levels=(
                (
                    np.percentile(new_view, self.real_space_autoscale_percentiles[0]),
                    np.percentile(new_view, self.real_space_autoscale_percentiles[1]),
                )
                if auto_level else None
            ),
            autoRange=reset,
        )
    else:
        new_view = complex_to_Lab(vimg)
        self.real_space_widget.setImage(
            np.transpose(new_view, (1, 0, 2)),
            autoLevels=False, levels=(0, 1), autoRange=reset,
        )

    # Statistics
    stats = [
        f"Min:\t{vimg.min():.5g}",
        f"Max:\t{vimg.max():.5g}",
        f"Mean:\t{vimg.mean():.5g}",
        f"Sum:\t{vimg.sum():.5g}",
        f"Std:\t{np.std(vimg):.5g}",
    ]
    for text, action in zip(stats, self.realspace_statistics_actions):
        action.setText(text)

    # FFT pane
    self.unscaled_fft_image = None
    vimg_2D = vimg if np.isrealobj(vimg) else np.abs(vimg)
    fft_window = (
        np.hanning(vimg_2D.shape[0])[:, None] * np.hanning(vimg_2D.shape[1])[None, :]
    )
    fft_label = self.fft_source_action_group.checkedAction().text()

    if fft_label == "Virtual Image FFT":
        fft = np.abs(np.fft.fftshift(np.fft.fft2(vimg_2D * fft_window))) ** 0.5
        levels = (np.min(fft), np.percentile(fft, 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.fft_widget_text.setText("Virtual Image FFT")
        self.fft_widget.setImage(fft.T, autoLevels=False, levels=levels, autoRange=mode_switch)
        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            self.fft_widget.autoRange()
        self.unscaled_fft_image = fft

    elif fft_label == "Virtual Image FFT (complex)":
        fft = np.fft.fftshift(np.fft.fft2(vimg_2D * fft_window))
        levels = (np.min(np.abs(fft)), np.percentile(np.abs(fft), 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.fft_widget_text.setText("Virtual Image FFT")
        fft_img = complex_to_Lab(fft.T, amin=levels[0], amax=levels[1], ab_scale=128, gamma=0.5)
        self.fft_widget.setImage(fft_img, autoLevels=False, autoRange=mode_switch, levels=(0, 1))
        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            self.fft_widget.autoRange()
        self.unscaled_fft_image = fft


# ---------------------------------------------------------------------------
# Diffraction-space update
# ---------------------------------------------------------------------------

def update_diffraction_space_view(self, reset=False):
    if self.datacube is None:
        return

    detector = self.get_virtual_image_detector()

    match detector["shape"]:
        case DetectorShape.POINT:
            xc, yc = detector["point"]
            self.real_space_view_text.setText(f"Virtual Image: Point [{xc},{yc}]")
            DP = self.datacube.data[xc, yc]

        case DetectorShape.RECTANGULAR:
            slice_x, slice_y = detector["slice"]
            self.real_space_view_text.setText(
                f"Virtual Image: Slice [{slice_x.start}:{slice_x.stop},"
                f"{slice_y.start}:{slice_y.stop}]"
            )
            match detector["mode"]:
                case DetectorMode.INTEGRATING:
                    DP = np.sum(self.datacube.data[slice_x, slice_y], axis=(0, 1))
                case DetectorMode.MAXIMUM:
                    DP = np.max(self.datacube.data[slice_x, slice_y], axis=(0, 1))
                case _:
                    raise ValueError("Unsupported detector mode for real-space selector.")

        case _:
            raise ValueError("Unsupported real-space detector shape.")

    self.set_diffraction_image(DP, reset=reset)


def set_diffraction_image(self, DP, reset=False):
    self.unscaled_diffraction_image = DP
    self._render_diffraction_image(reset=reset)


def _render_diffraction_image(self, reset=False):
    DP = self.unscaled_diffraction_image
    scaling_mode = self.diff_scaling_group.checkedAction().text().replace("&", "")

    if scaling_mode == "Linear":
        new_view = DP.copy()
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(DP, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(DP, 0))
    else:
        raise ValueError(f"Scaling mode not recognised: {scaling_mode!r}")

    stats = [
        f"Min:\t{DP.min():.5g}",
        f"Max:\t{DP.max():.5g}",
        f"Mean:\t{DP.mean():.5g}",
        f"Sum:\t{DP.sum():.5g}",
        f"Std:\t{np.std(DP):.5g}",
    ]
    for text, action in zip(stats, self.diffraction_statistics_actions):
        action.setText(text)

    auto_level = reset or self.diffraction_rescale_button.latched
    self.diffraction_space_widget.setImage(
        new_view.T,
        autoLevels=False,
        levels=(
            (
                np.percentile(new_view, self.diffraction_autoscale_percentiles[0]),
                np.percentile(new_view, self.diffraction_autoscale_percentiles[1]),
            )
            if auto_level else None
        ),
        autoRange=reset,
    )

    if self.fft_source_action_group.checkedAction().text() == "EWPC":
        log_clip = np.maximum(1e-10, np.percentile(np.maximum(DP, 0.0), 0.1))
        fft = np.abs(np.fft.fftshift(np.fft.fft2(np.log(np.maximum(DP, log_clip)))))
        levels = (np.min(fft), np.percentile(fft, 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "EWPC"
        self.fft_widget_text.setText("EWPC")
        self.fft_widget.setImage(fft.T, autoLevels=False, levels=levels, autoRange=mode_switch)


# ---------------------------------------------------------------------------
# Detector ROI management
# ---------------------------------------------------------------------------

def update_realspace_detector(self):
    shape = self.rs_detector_shape_group.checkedAction().text().replace("&", "")
    assert shape in ("Point", "Rectangular"), shape

    main_pen   = {"color": "g", "width": 6}
    handle_pen = {"color": "r", "width": 9}
    hover_pen  = {"color": "c", "width": 6}
    hover_hpen = {"color": "c", "width": 9}

    if self.datacube is None:
        x0, y0, xr, yr = 0, 0, 4, 4
    else:
        x, y = self.datacube.data.shape[2:]
        y0, x0 = x // 2, y // 2
        xr = yr = np.minimum(x, y) / 10

    if hasattr(self, "real_space_point_selector") and self.real_space_point_selector:
        self.real_space_widget.view.scene().removeItem(self.real_space_point_selector)
        self.real_space_point_selector = None
    if hasattr(self, "real_space_rect_selector") and self.real_space_rect_selector:
        self.real_space_widget.view.scene().removeItem(self.real_space_rect_selector)
        self.real_space_rect_selector = None

    if shape == "Point":
        self.real_space_point_selector = pg_point_roi(
            self.real_space_widget.getView(),
            center=(x0 - 0.5, y0 - 0.5),
            pen=main_pen, hoverPen=hover_pen,
        )
        self.real_space_point_selector.sigRegionChanged.connect(
            partial(self.update_diffraction_space_view, False)
        )
    elif shape == "Rectangular":
        self.real_space_rect_selector = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)],
            pen=main_pen, handlePen=handle_pen,
            hoverPen=hover_pen, handleHoverPen=hover_hpen,
        )
        self.real_space_widget.getView().addItem(self.real_space_rect_selector)
        self.real_space_rect_selector.sigRegionChangeFinished.connect(
            partial(self.update_diffraction_space_view, False)
        )

    self.update_diffraction_space_view(reset=True)


def update_diffraction_detector(self):
    shape = self.detector_shape_group.checkedAction().text().strip("&")
    assert shape in ("Point", "Rectangular", "Circle", "Annulus"), shape

    main_pen   = {"color": "g", "width": 6}
    handle_pen = {"color": "r", "width": 9}
    hover_pen  = {"color": "c", "width": 6}
    hover_hpen = {"color": "c", "width": 9}

    if self.datacube is None:
        x0, y0, xr, yr = 0, 0, 4, 4
    else:
        x, y = self.datacube.data.shape[2:]
        y0, x0 = x // 2, y // 2
        xr = yr = np.minimum(x, y) / 10

    for attr in (
        "virtual_detector_point", "virtual_detector_roi",
        "virtual_detector_roi_inner", "virtual_detector_roi_outer",
    ):
        item = getattr(self, attr, None)
        if item is not None:
            self.diffraction_space_widget.view.scene().removeItem(item)
            setattr(self, attr, None)

    if shape == "Point":
        self.virtual_detector_point = pg_point_roi(
            self.diffraction_space_widget.getView(),
            center=(x0 - 0.5, y0 - 0.5),
            pen=main_pen, hoverPen=hover_pen,
        )
        self.virtual_detector_point.sigRegionChanged.connect(
            partial(self.update_real_space_view, False)
        )

    elif shape == "Rectangular":
        self.virtual_detector_roi = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)],
            pen=main_pen, handlePen=handle_pen,
            hoverPen=hover_pen, handleHoverPen=hover_hpen,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    elif shape == "Circle":
        self.virtual_detector_roi = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)],
            pen=main_pen, handlePen=handle_pen,
            hoverPen=hover_pen, handleHoverPen=hover_hpen,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    elif shape == "Annulus":
        self.virtual_detector_roi_outer = pg.CircleROI(
            [int(x0 - xr), int(y0 - yr)], [int(2 * xr), int(2 * yr)],
            pen=main_pen, handlePen=handle_pen,
            hoverPen=hover_pen, handleHoverPen=hover_hpen,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_outer)

        self.virtual_detector_roi_inner = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)],
            pen=main_pen, hoverPen=hover_pen,
            handlePen=handle_pen, handleHoverPen=hover_hpen,
            movable=False,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_inner)

        self.virtual_detector_roi_outer.sigRegionChanged.connect(self.update_annulus_pos)
        self.virtual_detector_roi_outer.sigRegionChanged.connect(self.update_annulus_radii)
        self.virtual_detector_roi_inner.sigRegionChanged.connect(self.update_annulus_radii)
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )
        self.virtual_detector_roi_inner.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    self.update_real_space_view(reset=True)


def update_annulus_pos(self):
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
    y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
    self.virtual_detector_roi_inner.setPos(x0 - R_inner, y0 - R_inner, update=False)


def update_annulus_radii(self):
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    if R_outer < R_inner:
        x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
        y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
        self.virtual_detector_roi_outer.setSize(2 * R_inner + 6, update=False)
        self.virtual_detector_roi_outer.setPos(
            x0 - R_inner - 3, y0 - R_inner - 3, update=False
        )


# ---------------------------------------------------------------------------
# Autoscale range setters
# ---------------------------------------------------------------------------

def set_diffraction_autoscale_range(self, percentiles, redraw=True):
    self.diffraction_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/diffraction_autorange", list(percentiles))
    if redraw:
        self._render_diffraction_image(reset=False)


def set_real_space_autoscale_range(self, percentiles, redraw=True):
    self.real_space_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/realspace_autorange", list(percentiles))
    if redraw:
        self._render_virtual_image(reset=False)


# ---------------------------------------------------------------------------
# Keyboard nudge
# ---------------------------------------------------------------------------

def nudge_real_space_selector(self, dx, dy):
    selector = getattr(self, "real_space_point_selector", None) or getattr(
        self, "real_space_rect_selector", None
    )
    if selector is None:
        return
    pos = selector.pos()
    pos[0] += dy
    pos[1] += dx
    selector.setPos(pos)


def nudge_diffraction_selector(self, dx, dy):
    selector = (
        getattr(self, "virtual_detector_point", None)
        or getattr(self, "virtual_detector_roi", None)
        or getattr(self, "virtual_detector_roi_outer", None)
    )
    if selector is None:
        return
    pos = selector.pos()
    pos[0] += dy
    pos[1] += dx
    selector.setPos(pos)


# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

def update_tooltip(self):
    if self.datacube is None or not self.isActiveWindow():
        return

    for scene, data in [
        (self.diffraction_space_widget, self.unscaled_diffraction_image),
        (self.real_space_widget,        self.unscaled_realspace_image),
        (self.fft_widget,               self.unscaled_fft_image),
    ]:
        if data is None:
            continue
        pos_in_scene = scene.mapFromGlobal(QCursor.pos())
        if scene.getView().rect().contains(pos_in_scene):
            pos_in_data = scene.view.mapSceneToView(pos_in_scene)
            y = int(np.clip(np.floor(pos_in_data.x()), 0, data.shape[1] - 1))
            x = int(np.clip(np.floor(pos_in_data.y()), 0, data.shape[0] - 1))
            if np.isrealobj(data):
                self.cursor_value_text.setText(f"[{x},{y}]: {data[x, y]:.5g}")
            else:
                self.cursor_value_text.setText(
                    f"[{x},{y}]: |z|={np.abs(data[x, y]):.5g}, "
                    f"φ={np.degrees(np.angle(data[x, y])):.5g}°"
                )
