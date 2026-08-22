"""
QApplication setup and initialization for the Auto OpenMatte GUI.
"""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from gui import __app_name__, __version__
from gui.main_window import MainWindow
from gui.resources.styles import DARK_THEME_QSS


def create_application(argv: list) -> QApplication:
    """Create and configure the QApplication instance.

    Args:
        argv: Command-line arguments (typically sys.argv).

    Returns:
        Configured QApplication instance.
    """
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AutoOpenMatte")

    # Apply dark theme
    app.setStyleSheet(DARK_THEME_QSS)

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    return app


def run(app: QApplication) -> int:
    """Show the main window and start the event loop.

    Args:
        app: The QApplication instance.

    Returns:
        Application exit code.
    """
    window = MainWindow()
    window.show()
    return app.exec()
