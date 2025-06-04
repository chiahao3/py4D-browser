import os

import matplotlib
from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import (
    CustomLogger,
    print_system_info,
    set_accelerator,
    set_gpu_device,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QWidget


class PtyRADPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "chiahao3.ptyrad_plugin"

    # optional flags

    # Plugins may add a top-level menu on their own, or can opt to have
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = (
        True  # Put it under Plugins feels cleaner although it's one more level
    )
    display_name = "PtyRAD"

    def __init__(self, parent, plugin_menu, **kwargs):
        super().__init__()
        self.parent = parent

        # Add PtyRAD menu
        self.ptyrad_menu = plugin_menu

        # Add "Set Working Directory" option
        self.set_working_dir_action = QAction("Set PtyRAD Working Directory", self)
        self.set_working_dir_action.triggered.connect(self.set_ptyrad_working_directory)
        self.ptyrad_menu.addAction(self.set_working_dir_action)

        # Add "Import Params" option
        self.import_params_action = QAction("Import PtyRAD Params", self)
        self.import_params_action.triggered.connect(self.import_ptyrad_params)
        self.ptyrad_menu.addAction(self.import_params_action)

        # Add "Create Params" option
        self.create_params_action = QAction("Create PtyRAD Params", self)
        self.create_params_action.triggered.connect(self.create_ptyrad_params)
        self.ptyrad_menu.addAction(self.create_params_action)

        # Add "Run PtyRAD" option
        self.run_ptyrad_action = QAction("Run PtyRAD", self)
        self.run_ptyrad_action.triggered.connect(self.run_ptyrad)
        self.ptyrad_menu.addAction(self.run_ptyrad_action)

        # Add "Stop PtyRAD" button
        self.stop_ptyrad_action = QAction("Stop PtyRAD", self)
        self.stop_ptyrad_action.triggered.connect(self.stop_ptyrad)
        self.stop_ptyrad_action.setEnabled(False)  # Initially disabled
        self.ptyrad_menu.addAction(self.stop_ptyrad_action)

    def set_ptyrad_working_directory(self):
        # Open a directory selection dialog
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select PtyRAD Working Directory",
            os.getcwd(),  # Default to the current working directory
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if directory:
            try:
                # Change the working directory
                os.chdir(directory)
                self.ptyrad_working_directory = (
                    directory  # Store the selected directory
                )
                self.parent.statusBar().showMessage(
                    f"PtyRAD Working directory set to: {directory}", 5000
                )
                print(f"PtyRAD Working directory set to: {directory}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to set working directory: {str(e)}"
                )
        else:
            self.parent.statusBar().showMessage("No directory selected.", 5000)

    def import_ptyrad_params(self):
        print("Import PtyRAD Params selected.")

        # Open a file dialog to select a PtyRAD Params file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PtyRAD Parameters File",
            "",
            "PtyRAD Params Files (*.yaml *.yml *.json *.toml);;All Files (*)",
            options=options,
        )

        if file_path:
            try:
                # Use PtyRAD's load_params function to read the YAML file
                params = load_params(file_path)
                self.ptyrad_params = params
                # print("Loaded parameters:", params)  # Debugging output
                self.parent.statusBar().showMessage(
                    f"Loaded parameters from {file_path}", 5000
                )
            except Exception as e:
                # Handle errors during parameter loading
                QMessageBox.critical(
                    self, "Error", f"Failed to load parameters: {str(e)}"
                )
        else:
            self.parent.statusBar().showMessage("No file selected.", 5000)

    def create_ptyrad_params(self):
        print("Create PtyRAD Params selected.")
        print("This method is not implemented yet.")

    def run_ptyrad(self):
        from copy import deepcopy

        print("Run PtyRAD selected.")

        # Check if a thread is already running
        if hasattr(self, "runner") and self.runner.isRunning():
            QMessageBox.warning(
                self, "Error", "A PtyRAD reconstruction is already running!"
            )
            return

        # Check if working directory is set
        if (
            not hasattr(self, "ptyrad_working_directory")
            or not self.ptyrad_working_directory
        ):
            response = QMessageBox.question(
                self,
                "Missing PtyRAD Working Directory",
                "No PtyRAD working directory is set. Would you like to set it now?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if response == QMessageBox.Yes:
                self.set_ptyrad_working_directory()
                # Check again after setting the working directory
                if (
                    not hasattr(self, "ptyrad_working_directory")
                    or not self.ptyrad_working_directory
                ):
                    QMessageBox.warning(
                        self, "Error", "Failed to set working directory!"
                    )
                    return
            else:
                self.parent.statusBar().showMessage(
                    "PtyRAD reconstruction canceled.", 5000
                )
                return

        # Check if parameters are loaded
        if not hasattr(self, "ptyrad_params") or self.ptyrad_params is None:
            response = QMessageBox.question(
                self,
                "Missing PtyRAD Parameters",
                "No PtyRAD parameters are loaded. Would you like to import them now?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if response == QMessageBox.Yes:
                self.import_ptyrad_params()
                # Check again after importing
                if not hasattr(self, "ptyrad_params") or self.ptyrad_params is None:
                    QMessageBox.warning(self, "Error", "Failed to load parameters!")
                    return
            else:
                self.parent.statusBar().showMessage(
                    "PtyRAD reconstruction canceled.", 5000
                )
                return

        # Initialize the thread
        self.runner = PtyRADRunner(
            deepcopy(self.ptyrad_params)
        )  # Use deepcopy for thread safety
        self.runner.started.connect(self.on_ptyrad_started)
        self.runner.finished.connect(self.on_ptyrad_finished)
        self.runner.error.connect(self.on_ptyrad_error)

        # Start the thread
        self.parent.statusBar().showMessage("Running PtyRAD reconstruction...")
        self.runner.start()

    def stop_ptyrad(self):
        if hasattr(self, "runner") and self.runner.isRunning():
            self.runner.interrupt()  # Call the interrupt method in PtyRADRunner
            self.parent.statusBar().showMessage(
                "Stopping PtyRAD reconstruction...", 5000
            )
        else:
            QMessageBox.warning(
                self, "Error", "No active PtyRAD reconstruction to cancel!"
            )

    def on_ptyrad_started(self):
        # Disable actions while reconstruction is running
        self.set_working_dir_action.setEnabled(False)
        self.import_params_action.setEnabled(False)
        self.run_ptyrad_action.setEnabled(False)
        self.stop_ptyrad_action.setEnabled(True)  # Enable stop button

    def on_ptyrad_finished(self):
        self.runner = None
        # Re-enable actions after reconstruction finishes
        self.set_working_dir_action.setEnabled(True)
        self.import_params_action.setEnabled(True)
        self.run_ptyrad_action.setEnabled(True)
        self.stop_ptyrad_action.setEnabled(False)  # Disable stop button
        self.parent.statusBar().showMessage("PtyRAD reconstruction completed!", 5000)

    def on_ptyrad_error(self, error_message):
        self.runner = None
        QMessageBox.critical(
            self, "Error", f"PtyRAD reconstruction failed: {error_message}"
        )
        self.set_working_dir_action.setEnabled(True)
        self.import_params_action.setEnabled(True)
        self.run_ptyrad_action.setEnabled(True)
        self.stop_ptyrad_action.setEnabled(False)  # Disable stop button


class PtyRADRunner(QThread):
    finished = pyqtSignal()  # Signal emitted when the process finishes
    error = pyqtSignal(str)  # Signal emitted if an error occurs

    def __init__(self, ptyrad_params, parent=None):
        super().__init__(parent)
        self.ptyrad_params = ptyrad_params  # Store parameters for later use
        self.interrupted = False  # Flag for interruption
        self.original_backend = (
            matplotlib.get_backend()
        )  # Save the original matplotlib backend
        print(f"Original Matplotlib backend: {self.original_backend}")

    def interrupt(self):
        self.interrupted = True
        print("PtyRAD reconstruction interrupted by user.")

    def run(self):
        matplotlib.use(
            "Agg"
        )  # Set non-GUI backend to prevent warning, note that this is a global setting so it may affect other part of the py4DGUI
        print("Switch Matplotlib backend to 'Agg' for non-GUI figure saving")

        try:
            # Setup CustomLogger, currently hard code all the flags
            logger = CustomLogger(
                log_file="ptyrad_log.txt",
                log_dir="auto",
                prefix_date=True,
                prefix_jobid=0,
                append_to_file=True,
                show_timestamp=True,
                enable_terminal=True,
            )

            # Initialize PtyRAD components
            accelerator = set_accelerator()
            print_system_info()
            device = set_gpu_device(0) # Hardcode to GPU0 since PtyRAD automatically fallback to CPU if GPU not available

            # Initialize PtyRADSolver
            ptycho_solver = PtyRADSolver(
                self.ptyrad_params, device=device, acc=accelerator, logger=logger
            )

            # Pass the interrupt flag to PtyRADSolver
            ptycho_solver.interrupt = (
                lambda: self.interrupted
            )  # Pass a callable that returns the flag

            # Run PtyRAD reconstruction
            ptycho_solver.run()

            # Emit finished signal
            self.finished.emit()

        except Exception as e:
            # Handle other exceptions
            self.error.emit(str(e))

        finally:
            # Restore the original backend
            matplotlib.use(self.original_backend)
            print(f"Switch Matplotlib backend back to {self.original_backend}")
