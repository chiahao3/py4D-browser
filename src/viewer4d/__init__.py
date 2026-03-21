"""
viewer4d — Fast 4D-STEM data browser.

Forked from py4D-browser (Steven Zeltmann, LBL).
Rewritten to use PyTorch for GPU-accelerated virtual imaging,
tcBF, acBF, and iCOM, with rosettasciio for file I/O.
"""

from viewer4d.datacube import DataCube, Calibration

__version__ = "0.1.0"
__all__ = ["DataCube", "Calibration"]
