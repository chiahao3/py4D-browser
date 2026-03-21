"""
Main application window for viewer4d.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
import os
import sys
import platformdirs

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMenu, QAction, QHBoxLayout,
    QSplitter, QActionGroup, QLabel, QToolTip, QPushButton, QShortcut,
)

import pyqtgraph as pg
import numpy as np

from viewer4d.utils import pg_point_roi, VLine, LatchingButton
from viewer4d.scalebar import ScaleBar


class DataViewer(QMainWindow):
    """
    4D-STEM data browser main window.

    Instantiate and enter the Qt event loop::

        app = DataViewer(sys.argv)
        app.exec_()
    """

    LOG_SCALE_MIN_VALUE = 1e-6

    # ---------- mix-in methods from separate modules ----------
    from viewer4d.menu_actions import (
        load_file,
        load_data_arina,
        load_data_auto,
        load_data_bin,
        load_data_mmap,
        show_file_dialog,
        get_savefile_name,
        export_datacube,
        export_virtual_image,
        show_keyboard_map,
        reshape_data,
        set_datacube,
        update_scalebars,
    )

    from viewer4d.update_views import (
        set_virtual_image,
        set_diffraction_image,
        get_diffraction_detector,
        get_virtual_image_detector,
        _render_virtual_image,
        _render_diffraction_image,
        update_diffraction_space_view,
        update_real_space_view,
        update_realspace_detector,
        update_diffraction_detector,
        set_diffraction_autoscale_range,
        set_real_space_autoscale_range,
        nudge_real_space_selector,
        nudge_diffraction_selector,
        update_annulus_pos,
        update_annulus_radii,
        update_tooltip,
    )

    from viewer4d.plugins import load_plugins

    def __init__(self, argv):
        super().__init__()
        self.qtapp = QApplication.instance() or QApplication(argv)

        # Window chrome
        self.setWindowTitle("4D-Viewer")
        icon_path = Path(__file__).parent / "logo.png"
        if icon_path.exists():
            icon = QtGui.QIcon(str(icon_path))
            self.setWindowIcon(icon)
            self.qtapp.setWindowIcon(icon)

        self.setAcceptDrops(True)
        self.datacube = None

        # Persistent settings
        config_path = os.path.join(
            platformdirs.user_config_dir("viewer4d", "viewer4d"), "GUI_config.ini"
        )
        print(f"Config: {config_path}")
        QtCore.QCoreApplication.setOrganizationName("viewer4d")
        QtCore.QCoreApplication.setOrganizationDomain("viewer4d")
        QtCore.QCoreApplication.setApplicationName("viewer4d")
        self.settings = QtCore.QSettings(config_path, QtCore.QSettings.Format.IniFormat)

        if os.environ.get("VIEWER4D_RESET"):
            self.settings.remove("last_state")
            print("Cleared saved state.")

        self.setup_menus()
        self.setup_views()

        # Tooltip timer (30 Hz)
        self.tooltip_timer = pg.ThreadsafeTimer()
        self.tooltip_timer.timeout.connect(self.update_tooltip)
        self.tooltip_timer.start(1000 // 30)
        font = QtGui.QFont(self.font())
        font.setPointSize(10)
        QToolTip.setFont(font)

        self.resize(
            self.settings.value("last_state/window_size", QtCore.QSize(1200, 900))
        )

        self.load_plugins()
        self.show()

        if len(argv) > 1:
            self.load_file(argv[1])

        if os.environ.get("VIEWER4D_DEBUG"):
            pg.dbg()

    # ------------------------------------------------------------------
    # Menu setup
    # ------------------------------------------------------------------

    def setup_menus(self):
        self.menu_bar = self.menuBar()

        # ---- File ----
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        _lbl = QAction("Import", self)
        _lbl.setDisabled(True)
        self.file_menu.addAction(_lbl)

        self.load_auto_action = QAction("&Load Data...", self)
        self.load_auto_action.triggered.connect(self.load_data_auto)
        self.load_auto_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.file_menu.addAction(self.load_auto_action)

        self.load_mmap_action = QAction("Load &Memory Map...", self)
        self.load_mmap_action.triggered.connect(self.load_data_mmap)
        self.file_menu.addAction(self.load_mmap_action)

        self.load_binned_action = QAction("Load Data &Binned (4×)...", self)
        self.load_binned_action.triggered.connect(self.load_data_bin)
        self.file_menu.addAction(self.load_binned_action)

        self.load_arina_action = QAction("Load &Arina / Dectris Data...", self)
        self.load_arina_action.triggered.connect(self.load_data_arina)
        self.file_menu.addAction(self.load_arina_action)

        self.reshape_data_action = QAction("&Reshape Data...", self)
        self.reshape_data_action.triggered.connect(self.reshape_data)
        self.file_menu.addAction(self.reshape_data_action)

        self.file_menu.addSeparator()

        _lbl2 = QAction("Export", self)
        _lbl2.setDisabled(True)
        self.file_menu.addAction(_lbl2)

        dc_export_menu = QMenu("Export Datacube", self)
        self.file_menu.addMenu(dc_export_menu)
        for method in ["Raw float32", "HDF5"]:
            item = dc_export_menu.addAction(method)
            item.triggered.connect(partial(self.export_datacube, method))
            if method == "HDF5":
                item.setShortcut(QtGui.QKeySequence("Ctrl+S"))

        vimg_export_menu = QMenu("Export Virtual Image", self)
        self.file_menu.addMenu(vimg_export_menu)
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            item = vimg_export_menu.addAction(method)
            item.triggered.connect(partial(self.export_virtual_image, method, "image"))

        vdiff_export_menu = QMenu("Export Diffraction Pattern", self)
        self.file_menu.addMenu(vdiff_export_menu)
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            item = vdiff_export_menu.addAction(method)
            item.triggered.connect(
                partial(self.export_virtual_image, method, "diffraction")
            )

        # ---- Scaling ----
        self.scaling_menu = QMenu("&Scaling", self)
        self.menu_bar.addMenu(self.scaling_menu)

        diff_scaling_group = QActionGroup(self)
        diff_scaling_group.setExclusive(True)
        self.diff_scaling_group = diff_scaling_group
        _sep = QAction("Diffraction", self)
        _sep.setDisabled(True)
        self.scaling_menu.addAction(_sep)
        for label, checked in [("Linear", False), ("Log", False), ("Square Root", True)]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(partial(self._render_diffraction_image, True))
            diff_scaling_group.addAction(a)
            self.scaling_menu.addAction(a)

        self.scaling_menu.addSeparator()

        vimg_scaling_group = QActionGroup(self)
        vimg_scaling_group.setExclusive(True)
        self.vimg_scaling_group = vimg_scaling_group
        _sep2 = QAction("Virtual Image", self)
        _sep2.setDisabled(True)
        self.scaling_menu.addAction(_sep2)
        for label, checked in [("Linear", True), ("Log", False), ("Square Root", False)]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(partial(self._render_virtual_image, True))
            vimg_scaling_group.addAction(a)
            self.scaling_menu.addAction(a)
            if label == "Linear":
                self.vimg_scale_linear_action = a

        # ---- Autorange ----
        self.autorange_menu = QMenu("&Autorange", self)
        self.menu_bar.addMenu(self.autorange_menu)

        _sep3 = QAction("Diffraction", self)
        _sep3.setDisabled(True)
        self.autorange_menu.addAction(_sep3)
        diff_range_group = QActionGroup(self)
        diff_range_group.setExclusive(True)
        default_diff = self.settings.value(
            "last_state/diffraction_autorange", [0.1, 99.9], type=float
        )
        for r in [(0, 100), (0.1, 99.9), (1, 99), (2, 98), (5, 95)]:
            a = QAction(f"{r[0]}% – {r[1]}%", self)
            a.setCheckable(True)
            a.triggered.connect(partial(self.set_diffraction_autoscale_range, r))
            diff_range_group.addAction(a)
            self.autorange_menu.addAction(a)
            if r[0] == default_diff[0] and r[1] == default_diff[1]:
                a.setChecked(True)
                self.set_diffraction_autoscale_range(r, redraw=False)

        self.autorange_menu.addSeparator()

        _sep4 = QAction("Virtual Image", self)
        _sep4.setDisabled(True)
        self.autorange_menu.addAction(_sep4)
        vimg_range_group = QActionGroup(self)
        vimg_range_group.setExclusive(True)
        default_vimg = self.settings.value(
            "last_state/realspace_autorange", [0.1, 99.9], type=float
        )
        for r in [(0, 100), (0.1, 99.9), (1, 99), (2, 98), (5, 95)]:
            a = QAction(f"{r[0]}% – {r[1]}%", self)
            a.setCheckable(True)
            a.triggered.connect(partial(self.set_real_space_autoscale_range, r))
            vimg_range_group.addAction(a)
            self.autorange_menu.addAction(a)
            if r[0] == default_vimg[0] and r[1] == default_vimg[1]:
                a.setChecked(True)
                self.set_real_space_autoscale_range(r, redraw=False)

        # ---- Detector Response ----
        self.detector_menu = QMenu("&Detector Response", self)
        self.menu_bar.addMenu(self.detector_menu)

        _sep5 = QAction("Diffraction", self)
        _sep5.setDisabled(True)
        self.detector_menu.addAction(_sep5)
        detector_mode_group = QActionGroup(self)
        detector_mode_group.setExclusive(True)
        self.detector_mode_group = detector_mode_group
        for label, checked in [
            ("&Integrating", True), ("&Maximum", False), ("C&oM", False),
            ("CoM &X", False), ("CoM &Y", False), ("i&CoM", False),
        ]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(partial(self.update_real_space_view, True))
            detector_mode_group.addAction(a)
            self.detector_menu.addAction(a)

        self.detector_menu.addSeparator()
        _sep6 = QAction("Virtual Image", self)
        _sep6.setDisabled(True)
        self.detector_menu.addAction(_sep6)
        rs_detector_mode_group = QActionGroup(self)
        rs_detector_mode_group.setExclusive(True)
        self.realspace_detector_mode_group = rs_detector_mode_group
        for label, checked in [("&Integrating", True), ("&Maximum", False)]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(partial(self.update_diffraction_space_view, True))
            rs_detector_mode_group.addAction(a)
            self.detector_menu.addAction(a)

        # ---- Detector Shape ----
        self.detector_shape_menu = QMenu("Detector &Shape", self)
        self.menu_bar.addMenu(self.detector_shape_menu)

        detector_shape_group = QActionGroup(self)
        detector_shape_group.setExclusive(True)
        self.detector_shape_group = detector_shape_group
        _sep7 = QAction("Diffraction", self)
        _sep7.setDisabled(True)
        self.detector_shape_menu.addAction(_sep7)
        for label, checked in [
            ("&Point", True), ("&Rectangular", False),
            ("&Circle", False), ("&Annulus", False),
        ]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(self.update_diffraction_detector)
            detector_shape_group.addAction(a)
            self.detector_shape_menu.addAction(a)

        self.detector_shape_menu.addSeparator()
        _sep8 = QAction("Virtual Image", self)
        _sep8.setDisabled(True)
        self.detector_shape_menu.addAction(_sep8)
        rs_detector_shape_group = QActionGroup(self)
        rs_detector_shape_group.setExclusive(True)
        self.rs_detector_shape_group = rs_detector_shape_group
        for label, checked in [("Poin&t", True), ("Rectan&gular", False)]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(self.update_realspace_detector)
            rs_detector_shape_group.addAction(a)
            self.detector_shape_menu.addAction(a)

        # ---- FFT View ----
        self.fft_menu = QMenu("FF&T View", self)
        self.menu_bar.addMenu(self.fft_menu)
        self.fft_source_action_group = QActionGroup(self)
        self.fft_source_action_group.setExclusive(True)
        for label, checked, callback in [
            ("Virtual Image FFT",          True,  partial(self.update_real_space_view, False)),
            ("Virtual Image FFT (complex)", False, partial(self.update_real_space_view, False)),
            ("EWPC",                       False, partial(self.update_diffraction_space_view, False)),
        ]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(callback)
            self.fft_source_action_group.addAction(a)
            self.fft_menu.addAction(a)

        # ---- Plugins ----
        self.processing_menu = QMenu("&Plugins", self)
        self.menu_bar.addMenu(self.processing_menu)

        # ---- Tools (PtyRAD etc.) ----
        self.tools_menu = QMenu("&Tools", self)
        self.menu_bar.addMenu(self.tools_menu)

        # ---- Help ----
        self.help_menu = QMenu("&Help", self)
        self.menu_bar.addMenu(self.help_menu)
        km_action = QAction("Show &Keyboard Map", self)
        km_action.triggered.connect(self.show_keyboard_map)
        self.help_menu.addAction(km_action)

    # ------------------------------------------------------------------
    # View setup
    # ------------------------------------------------------------------

    def setup_views(self):
        # Diffraction panel
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((512, 512)))
        self.diffraction_space_widget.setMouseTracking(True)
        self.update_diffraction_detector()

        self.diffraction_scale_bar = ScaleBar(pixel_size=1, units="px", width=10)
        self.diffraction_scale_bar.setParentItem(self.diffraction_space_widget.getView())
        self.diffraction_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))
        self.diffraction_space_widget.setWindowTitle("Diffraction Space")

        # Virtual image panel
        self.real_space_widget = pg.ImageView()
        self.real_space_widget.setImage(np.zeros((512, 512)))
        self.update_realspace_detector()

        self.real_space_scale_bar = ScaleBar(pixel_size=1, units="px", width=10)
        self.real_space_scale_bar.setParentItem(self.real_space_widget.getView())
        self.real_space_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))
        self.real_space_widget.setWindowTitle("Virtual Image")

        # Drag-and-drop support
        for widget in (
            self.diffraction_space_widget, self.real_space_widget
        ):
            widget.setAcceptDrops(True)
            widget.dragEnterEvent = self.dragEnterEvent
            widget.dropEvent = self.dropEvent

        # FFT panel
        self.fft_widget = pg.ImageView()
        self.fft_widget.setImage(np.zeros((512, 512)))
        self.fft_scale_bar = ScaleBar(pixel_size=1, units="1/px", width=10)
        self.fft_scale_bar.setParentItem(self.fft_widget.getView())
        self.fft_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))
        self.fft_widget.setWindowTitle("FFT / EWPC")
        self.fft_widget_text = pg.TextItem("FFT", (200, 200, 200), None, (0, 1))
        self.fft_widget.addItem(self.fft_widget_text)
        self.fft_widget.setAcceptDrops(True)
        self.fft_widget.dragEnterEvent = self.dragEnterEvent
        self.fft_widget.dropEvent = self.dropEvent

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.diffraction_space_widget, 1)

        right = QSplitter()
        right.addWidget(self.real_space_widget)
        right.addWidget(self.fft_widget)
        right.setOrientation(QtCore.Qt.Vertical)
        h = self.real_space_widget.size().height() + self.fft_widget.size().height()
        right.setSizes([int(h * 2 / 3), int(h / 3)])
        layout.addWidget(right, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        for w in (self.diffraction_space_widget, self.real_space_widget, self.fft_widget):
            w.getView().setMenuEnabled(False)

        # Status bar widgets
        self.stats_button = QPushButton("Statistics")
        self.stats_menu = QMenu()
        self.realspace_title = QAction("Virtual Image")
        self.realspace_title.setDisabled(False)
        self.stats_menu.addAction(self.realspace_title)
        self.realspace_statistics_actions = [QAction("") for _ in range(5)]
        for a in self.realspace_statistics_actions:
            self.stats_menu.addAction(a)
        self.stats_menu.addSeparator()
        self.diffraction_title = QAction("Diffraction")
        self.diffraction_title.setDisabled(False)
        self.stats_menu.addAction(self.diffraction_title)
        self.diffraction_statistics_actions = [QAction("") for _ in range(5)]
        for a in self.diffraction_statistics_actions:
            self.stats_menu.addAction(a)
        self.stats_button.setMenu(self.stats_menu)

        self.cursor_value_text = QLabel("")
        self.diffraction_space_view_text = QLabel("Slice")
        self.real_space_view_text = QLabel("Scan Position")

        sb = self.statusBar()
        sb.addPermanentWidget(self.cursor_value_text)
        sb.addPermanentWidget(VLine())
        sb.addPermanentWidget(self.stats_button)
        sb.addPermanentWidget(VLine())
        sb.addPermanentWidget(self.diffraction_space_view_text)
        sb.addPermanentWidget(VLine())
        sb.addPermanentWidget(self.real_space_view_text)
        sb.addPermanentWidget(VLine())

        self.diffraction_rescale_button = LatchingButton(
            "Autorange Diffraction", status_bar=sb, latched=True
        )
        self.diffraction_rescale_button.activated.connect(
            self.diffraction_space_widget.autoLevels
        )
        sb.addPermanentWidget(self.diffraction_rescale_button)

        self.realspace_rescale_button = LatchingButton(
            "Autorange Virtual Image", status_bar=sb, latched=True
        )
        self.realspace_rescale_button.activated.connect(
            self.real_space_widget.autoLevels
        )
        sb.addPermanentWidget(self.realspace_rescale_button)

        # Initialise unscaled image holders
        self.unscaled_realspace_image = np.zeros((1, 1))
        self.unscaled_diffraction_image = np.zeros((1, 1))
        self.unscaled_fft_image = None

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        self.settings.setValue("last_state/window_size", event.size())

    def dragEnterEvent(self, event):
        event.accept() if event.mimeData().hasUrls() else event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if len(files) == 1:
            self.load_file(files[0])

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()
        speed = 5 if modifier == QtCore.Qt.ShiftModifier else 1

        if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D):
            self.nudge_diffraction_selector(
                dx=speed * (-1 if key == QtCore.Qt.Key_W else 1 if key == QtCore.Qt.Key_S else 0),
                dy=speed * (-1 if key == QtCore.Qt.Key_A else 1 if key == QtCore.Qt.Key_D else 0),
            )
        elif key in (QtCore.Qt.Key_I, QtCore.Qt.Key_J, QtCore.Qt.Key_K, QtCore.Qt.Key_L):
            self.nudge_real_space_selector(
                dx=speed * (-1 if key == QtCore.Qt.Key_I else 1 if key == QtCore.Qt.Key_K else 0),
                dy=speed * (-1 if key == QtCore.Qt.Key_J else 1 if key == QtCore.Qt.Key_L else 0),
            )
