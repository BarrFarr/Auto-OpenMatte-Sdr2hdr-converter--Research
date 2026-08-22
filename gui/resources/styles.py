"""
Dark theme QSS stylesheet for the Auto OpenMatte GUI.

Designed for HDR video workflow - dark background reduces eye strain
when working with HDR content on calibrated displays.
"""

DARK_THEME_QSS = """
/* ==================== Base ==================== */
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 10pt;
}

/* ==================== Tabs ==================== */
QTabWidget::pane {
    border: 1px solid #3a3a3a;
    background-color: #252525;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #b0b0b0;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #252525;
    color: #ffffff;
    border-bottom: 2px solid #4a90d9;
}

QTabBar::tab:hover {
    background-color: #353535;
    color: #ffffff;
}

/* ==================== Group Boxes ==================== */
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #4a90d9;
}

/* ==================== Buttons ==================== */
QPushButton {
    background-color: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 60px;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #5a5a5a;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
    border-color: #333333;
}

QPushButton[class="primary-button"] {
    background-color: #2d5a8c;
    border-color: #4a90d9;
    color: #ffffff;
}

QPushButton[class="primary-button"]:hover {
    background-color: #366fa8;
}

QPushButton[class="lock-button"] {
    background-color: #2d6b2d;
    border-color: #44cc44;
    color: #ffffff;
    font-weight: bold;
    font-size: 12pt;
}

QPushButton[class="lock-button"]:hover {
    background-color: #3a8a3a;
}

QPushButton[class="lock-button"]:disabled {
    background-color: #1a3a1a;
    color: #666666;
}

QPushButton[class="toggle-button"] {
    background-color: #5a3a8c;
    border-color: #8a5ad9;
}

/* ==================== Inputs ==================== */
QLineEdit, QSpinBox, QComboBox {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 4px 8px;
}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #4a90d9;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    color: #e0e0e0;
    selection-background-color: #4a90d9;
}

/* ==================== Progress Bar ==================== */
QProgressBar {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #4a90d9;
    border-radius: 3px;
}

/* ==================== Sliders ==================== */
QSlider::groove:horizontal {
    background-color: #3a3a3a;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #4a90d9;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #5aa0e9;
}

/* ==================== Tables ==================== */
QTableWidget {
    background-color: #252525;
    alternate-background-color: #2a2a2a;
    gridline-color: #3a3a3a;
    border: 1px solid #3a3a3a;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #2d5a8c;
}

QHeaderView::section {
    background-color: #2d2d2d;
    color: #b0b0b0;
    padding: 6px;
    border: 1px solid #3a3a3a;
    font-weight: bold;
}

/* ==================== Labels ==================== */
QLabel[class="field-name"] {
    color: #888888;
    font-weight: bold;
}

QLabel[class="field-value"] {
    color: #e0e0e0;
}

QLabel[class="hint-text"] {
    color: #666666;
    font-style: italic;
}

QLabel[class="offset-value"] {
    font-size: 14pt;
    font-weight: bold;
    color: #b0b0b0;
}

QLabel[class="offset-value-primary"] {
    font-size: 16pt;
    font-weight: bold;
    color: #4a90d9;
}

QLabel[class="metric-name"] {
    color: #b0b0b0;
}

QLabel[class="metric-value"] {
    font-family: "Consolas", "JetBrains Mono", monospace;
    font-size: 11pt;
}

QLabel[class="frame-display"] {
    background-color: #0a0a0a;
    border: 1px solid #3a3a3a;
    border-radius: 2px;
    color: #666666;
    font-size: 12pt;
}

QLabel[class="info-text"] {
    color: #888888;
    font-style: italic;
}

/* ==================== Frames ==================== */
QFrame[class="drop-zone"] {
    background-color: #252525;
    border: 2px dashed #4a4a4a;
    border-radius: 8px;
}

QFrame[class="drop-zone"]:hover {
    border-color: #4a90d9;
}

QFrame[class="current-offset"] {
    border: 2px solid #4a90d9;
    border-radius: 4px;
}

/* ==================== Status Bar ==================== */
QStatusBar {
    background-color: #252525;
    color: #b0b0b0;
    border-top: 1px solid #3a3a3a;
}

/* ==================== Menu Bar ==================== */
QMenuBar {
    background-color: #252525;
    color: #e0e0e0;
}

QMenuBar::item:selected {
    background-color: #3a3a3a;
}

QMenu {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
}

QMenu::item:selected {
    background-color: #4a90d9;
}

/* ==================== Scrollbars ==================== */
QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
}

QScrollBar::handle:vertical {
    background-color: #4a4a4a;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5a5a5a;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 12px;
}

QScrollBar::handle:horizontal {
    background-color: #4a4a4a;
    border-radius: 4px;
    min-width: 20px;
}

QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}

/* ==================== Splitter ==================== */
QSplitter::handle {
    background-color: #3a3a3a;
}

QSplitter::handle:hover {
    background-color: #4a90d9;
}
"""
