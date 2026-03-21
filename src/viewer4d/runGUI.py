"""Entry point for the 4D-Viewer GUI."""

import sys


def launch():
    from viewer4d.main_window import DataViewer

    app = DataViewer(sys.argv)
    sys.exit(app.qtapp.exec_())


if __name__ == "__main__":
    launch()
