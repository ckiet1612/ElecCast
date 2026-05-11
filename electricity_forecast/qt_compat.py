from __future__ import annotations

# ruff: noqa: F401

try:
    from PyQt6.QtCore import (
        QCoreApplication,
        QLibraryInfo,
        QObject,
        Qt,
        QThread,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt6"
    Signal = pyqtSignal
except ModuleNotFoundError:
    from PySide6.QtCore import (
        QCoreApplication,
        QLibraryInfo,
        QObject,
        Qt,
        QThread,
        QTimer,
        Signal,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"


def add_library_path() -> None:
    plugins_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if plugins_path:
        QCoreApplication.addLibraryPath(plugins_path)
