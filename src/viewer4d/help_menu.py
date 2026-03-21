"""Keyboard map dialog."""

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QDialog, QVBoxLayout
from pathlib import Path


class KeyboardMapMenu(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        keymap_path = Path(__file__).parent / "keymap.png"
        self.keymap = QtGui.QPixmap(str(keymap_path)).scaledToWidth(1500)

        label = _ScaledLabel()
        label.setPixmap(self.keymap)

        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.resize(self.keymap.width(), self.keymap.height())


class _ScaledLabel(QWidget):
    """Widget that displays a QPixmap scaled to fit, maintaining aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.p = QtGui.QPixmap()

    def setPixmap(self, p):
        self.p = p
        self.update()

    def paintEvent(self, event):
        if not self.p.isNull():
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            rect = self.rect()
            rect.setHeight(int(self.p.height() * rect.width() / self.p.width()))
            painter.drawPixmap(rect, self.p)
