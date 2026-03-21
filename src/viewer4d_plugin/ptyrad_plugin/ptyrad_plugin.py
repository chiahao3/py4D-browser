"""
PtyRAD launcher plugin.

Adds a "Tools > Launch PtyRAD…" menu item that opens a configuration
dialog for starting a ptychographic reconstruction with PtyRAD.

PtyRAD integration is designed to be flexible:
  - If PtyRAD is installed as a Python package (``import ptyrad``), its
    Python API is used directly in a worker thread.
  - If PtyRAD is only available as a CLI tool, a subprocess is spawned
    using the path specified in the dialog.
  - If neither is available, the dialog prints a helpful install message.

The dialog exposes the minimal set of parameters needed to bootstrap a
reconstruction from the currently loaded datacube:
  dataset path, scan shape, pixel sizes, detector distance, beam energy.
A YAML / TOML config file can also be loaded/edited directly.

This is a scaffold — wire up to PtyRAD's actual API when it stabilises.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QDialog, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QGroupBox, QGridLayout,
    QFileDialog, QTextEdit, QSplitter, QCheckBox, QComboBox,
)


def _ptyrad_available() -> bool:
    try:
        import ptyrad  # noqa: F401
        return True
    except ImportError:
        return False


class PtyRADPlugin(QWidget):
    """
    Registered in the Tools menu (not the Plugins menu) by main_window.py.
    """

    plugin_id    = "viewer4d.ptyrad"
    display_name = "Launch PtyRAD…"

    # Uses a single action registered under the Tools menu.
    # The main_window.py tools_menu picks this up via the special flag below.
    uses_tools_action = True
    uses_plugin_menu  = False
    uses_single_action = False   # we self-register in Tools, see __init__

    def __init__(self, parent, **kwargs):
        super().__init__()
        self.parent = parent

        # Add directly to the Tools menu rather than the Plugins menu
        tools_action = parent.tools_menu.addAction(self.display_name)
        tools_action.triggered.connect(self._launch)

        # Also expose a separator and about info
        parent.tools_menu.addSeparator()
        about = parent.tools_menu.addAction("About PtyRAD")
        about.triggered.connect(self._show_about)

    def close(self):
        pass

    def _launch(self):
        PtyRADDialog(parent=self.parent).show()

    def _show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        status = "installed ✓" if _ptyrad_available() else "not found"
        QMessageBox.information(
            self.parent,
            "About PtyRAD",
            f"PtyRAD — GPU-accelerated ptychographic reconstruction\n\n"
            f"Status: {status}\n\n"
            f"Install:  pip install ptyrad\n"
            f"GitHub:   https://github.com/chiahao3/ptyrad",
        )


class PtyRADDialog(QDialog):
    """
    Launch dialog for PtyRAD.

    Left panel  — parameter form populated from the loaded datacube.
    Right panel — log / output console.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.setWindowTitle("Launch PtyRAD")
        self.resize(900, 600)
        self._worker = None

        splitter = QSplitter(Qt.Horizontal, self)
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        # ---- Left: parameters ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        dc = parent.datacube

        # Dataset
        ds_box = QGroupBox("Dataset")
        left_layout.addWidget(ds_box)
        ds_grid = QGridLayout()
        ds_box.setLayout(ds_grid)

        ds_grid.addWidget(QLabel("File path"), 0, 0, Qt.AlignRight)
        self.filepath_box = QLineEdit(parent.windowTitle())
        ds_grid.addWidget(self.filepath_box, 0, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_file)
        ds_grid.addWidget(browse, 0, 2)

        ds_grid.addWidget(QLabel("Scan shape (Nx, Ny)"), 1, 0, Qt.AlignRight)
        self.scan_shape_box = QLineEdit(
            f"{dc.R_Nx}, {dc.R_Ny}" if dc else "256, 256"
        )
        ds_grid.addWidget(self.scan_shape_box, 1, 1, 1, 2)

        ds_grid.addWidget(QLabel("Det shape (Nx, Ny)"), 2, 0, Qt.AlignRight)
        self.det_shape_box = QLineEdit(
            f"{dc.Q_Nx}, {dc.Q_Ny}" if dc else "128, 128"
        )
        ds_grid.addWidget(self.det_shape_box, 2, 1, 1, 2)

        # Calibration
        cal_box = QGroupBox("Calibration")
        left_layout.addWidget(cal_box)
        cal_grid = QGridLayout()
        cal_box.setLayout(cal_grid)

        cal = dc.calibration if dc else None

        cal_grid.addWidget(QLabel("Beam energy [keV]"), 0, 0, Qt.AlignRight)
        self.energy_box = QLineEdit("300")
        cal_grid.addWidget(self.energy_box, 0, 1)

        cal_grid.addWidget(QLabel("Scan pixel [Å]"), 1, 0, Qt.AlignRight)
        r_hint = f"{cal.get_R_pixel_size():.4g}" if cal else ""
        self.r_pix_box = QLineEdit(r_hint)
        self.r_pix_box.setPlaceholderText("Å / scan pixel")
        cal_grid.addWidget(self.r_pix_box, 1, 1)

        cal_grid.addWidget(QLabel("Det pixel [mrad]"), 2, 0, Qt.AlignRight)
        q_hint = (
            f"{cal.get_Q_pixel_size():.4g}"
            if (cal and cal.get_Q_pixel_units() == "mrad") else ""
        )
        self.q_pix_box = QLineEdit(q_hint)
        self.q_pix_box.setPlaceholderText("mrad / det pixel")
        cal_grid.addWidget(self.q_pix_box, 2, 1)

        # Config file (optional override)
        cfg_box = QGroupBox("Config file (optional)")
        left_layout.addWidget(cfg_box)
        cfg_layout = QHBoxLayout()
        cfg_box.setLayout(cfg_layout)
        self.config_box = QLineEdit()
        self.config_box.setPlaceholderText("path/to/ptyrad_config.yaml")
        cfg_layout.addWidget(self.config_box)
        browse_cfg = QPushButton("…")
        browse_cfg.setFixedWidth(28)
        browse_cfg.clicked.connect(self._browse_config)
        cfg_layout.addWidget(browse_cfg)

        # Execution
        exec_box = QGroupBox("Execution")
        left_layout.addWidget(exec_box)
        exec_grid = QGridLayout()
        exec_box.setLayout(exec_grid)

        exec_grid.addWidget(QLabel("Mode"), 0, 0, Qt.AlignRight)
        self.mode_combo = QComboBox()
        modes = []
        if _ptyrad_available():
            modes.append("Python API (in-process)")
        modes.append("CLI subprocess")
        self.mode_combo.addItems(modes)
        exec_grid.addWidget(self.mode_combo, 0, 1)

        exec_grid.addWidget(QLabel("CLI command"), 1, 0, Qt.AlignRight)
        self.cli_box = QLineEdit("ptyrad")
        self.cli_box.setPlaceholderText("ptyrad  or  python -m ptyrad")
        exec_grid.addWidget(self.cli_box, 1, 1)

        left_layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.run_btn = QPushButton("Launch PtyRAD")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._run)
        btn_layout.addWidget(self.run_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        left_layout.addLayout(btn_layout)

        # ---- Right: log console ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Output"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.log.setFont(font)
        right_layout.addWidget(self.log)
        splitter.addWidget(right)
        splitter.setSizes([420, 460])

        self._log("viewer4d  →  PtyRAD launcher")
        if not _ptyrad_available():
            self._log(
                "\nPtyRAD not found in this environment.\n"
                "Install with:  pip install ptyrad\n"
                "Or use CLI subprocess mode if ptyrad is on PATH."
            )

    # ------------------------------------------------------------------

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset file")
        if path:
            self.filepath_box.setText(path)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PtyRAD config", "",
            "Config files (*.yaml *.yml *.toml *.json);;All files (*)"
        )
        if path:
            self.config_box.setText(path)

    def _log(self, text: str):
        self.log.append(text)

    def _run(self):
        mode = self.mode_combo.currentText()
        self._log(f"\n[{mode}] Starting reconstruction…\n")

        params = self._gather_params()
        self._log("Parameters:\n" + "\n".join(f"  {k}: {v}" for k, v in params.items()))

        if "Python API" in mode:
            self._run_api(params)
        else:
            self._run_cli(params)

    def _gather_params(self) -> dict:
        return {
            "filepath":   self.filepath_box.text(),
            "scan_shape": self.scan_shape_box.text(),
            "det_shape":  self.det_shape_box.text(),
            "energy_keV": self.energy_box.text(),
            "r_pix_A":    self.r_pix_box.text(),
            "q_pix_mrad": self.q_pix_box.text(),
            "config":     self.config_box.text() or None,
        }

    def _run_api(self, params: dict):
        try:
            import ptyrad
            self._log(f"PtyRAD version: {getattr(ptyrad, '__version__', 'unknown')}")
            # TODO: wire to ptyrad's actual API once it stabilises.
            # Likely something like:
            #   recon = ptyrad.Reconstruction(config=params["config"] or params)
            #   recon.run()
            self._log(
                "\n[stub] PtyRAD Python API integration is a scaffold.\n"
                "Edit viewer4d_plugin/ptyrad_plugin/ptyrad_plugin.py → _run_api()\n"
                "to wire in ptyrad's actual reconstruction API."
            )
        except Exception as exc:
            self._log(f"Error: {exc}")

    def _run_cli(self, params: dict):
        cmd = self.cli_box.text().split()
        if params["config"]:
            cmd += ["--config", params["config"]]
        else:
            # Write a minimal temp config
            config_text = self._build_minimal_config(params)
            tmp_path = Path("/tmp/ptyrad_viewer4d_config.yaml")
            tmp_path.write_text(config_text)
            cmd += ["--config", str(tmp_path)]
            self._log(f"Wrote temporary config to {tmp_path}")

        self._log(f"Running: {' '.join(cmd)}\n")
        self._worker = _SubprocessWorker(cmd)
        self._worker.output.connect(self._log)
        self._worker.finished.connect(lambda: self._log("\n[Done]"))
        self._worker.start()

    def _build_minimal_config(self, params: dict) -> str:
        return textwrap.dedent(f"""\
            # Minimal PtyRAD config generated by viewer4d
            dataset:
              filepath:   "{params['filepath']}"
              scan_shape: [{params['scan_shape']}]
              det_shape:  [{params['det_shape']}]

            beam:
              energy_keV: {params['energy_keV'] or 300}

            calibration:
              r_pixel_size_A:    {params['r_pix_A'] or 1.0}
              q_pixel_size_mrad: {params['q_pix_mrad'] or 1.0}
        """)


class _SubprocessWorker(QThread):
    output   = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, cmd: list[str]):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                self.output.emit(line.rstrip())
            proc.wait()
            self.output.emit(f"\nProcess exited with code {proc.returncode}")
        except FileNotFoundError:
            self.output.emit(
                f"Error: command not found: {self.cmd[0]!r}\n"
                "Check the CLI command field or install PtyRAD."
            )
        except Exception as exc:
            self.output.emit(f"Error: {exc}")
        finally:
            self.finished.emit()
