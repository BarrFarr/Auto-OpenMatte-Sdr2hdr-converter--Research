"""
Preview Panel - Dual-source display with multiple viewing modes.

Shows HDR and OpenMatte sources simultaneously with:
- 4 display modes: side-by-side, overlay, wiper, checkerboard
- Frame-accurate navigation: +/-1, +/-10, +/-1 second, timecode entry
- Zoom controls: Fit, 100%, zoom/pan
"""
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QSlider,
    QFrame,
    QSplitter,
    QToolBar,
    QLineEdit,
)
from PySide6.QtCore import Qt, Slot, Signal, QSize
from PySide6.QtGui import QPixmap, QPainter, QColor

from gui.models.state import AppState


class PreviewMode:
    """Constants for preview display modes."""

    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    WIPER = "wiper"
    CHECKERBOARD = "checkerboard"


class FrameDisplayWidget(QWidget):
    """Placeholder widget for frame display (actual rendering requires GPU)."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self._label = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.frame_label = QLabel(self)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setProperty("class", "frame-display")
        self.frame_label.setText(f"[{label}]\nNo frame loaded")
        self.frame_label.setMinimumSize(320, 180)
        layout.addWidget(self.frame_label)

    def set_frame_info(self, frame_num: int, source_name: str):
        """Update display with frame information."""
        self.frame_label.setText(
            f"[{self._label}]\n{source_name}\nFrame: {frame_num}"
        )


class PreviewPanel(QWidget):
    """Panel for dual-source preview with display modes and navigation."""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._current_mode = PreviewMode.SIDE_BY_SIDE
        self._zoom_level = 1.0

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar: display mode and zoom
        toolbar_layout = QHBoxLayout()

        # Display mode selector
        toolbar_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("Side by Side", PreviewMode.SIDE_BY_SIDE)
        self.mode_combo.addItem("Overlay", PreviewMode.OVERLAY)
        self.mode_combo.addItem("Wiper", PreviewMode.WIPER)
        self.mode_combo.addItem("Checkerboard", PreviewMode.CHECKERBOARD)
        toolbar_layout.addWidget(self.mode_combo)

        toolbar_layout.addSpacing(20)

        # Zoom controls
        toolbar_layout.addWidget(QLabel("Zoom:"))
        self.zoom_fit_btn = QPushButton("Fit", self)
        self.zoom_100_btn = QPushButton("100%", self)
        self.zoom_in_btn = QPushButton("+", self)
        self.zoom_out_btn = QPushButton("-", self)
        self.zoom_label = QLabel("100%", self)
        toolbar_layout.addWidget(self.zoom_fit_btn)
        toolbar_layout.addWidget(self.zoom_100_btn)
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.zoom_label)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # Frame display area
        display_frame = QFrame(self)
        display_frame.setFrameShape(QFrame.Shape.StyledPanel)
        display_layout = QHBoxLayout(display_frame)

        # Dual source displays (side-by-side by default)
        self.hdr_display = FrameDisplayWidget("HDR", self)
        self.om_display = FrameDisplayWidget("OpenMatte", self)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.hdr_display)
        self.splitter.addWidget(self.om_display)
        display_layout.addWidget(self.splitter)

        main_layout.addWidget(display_frame, 1)

        # Wiper slider (only visible in wiper mode)
        self.wiper_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.wiper_slider.setRange(0, 100)
        self.wiper_slider.setValue(50)
        self.wiper_slider.setVisible(False)
        main_layout.addWidget(self.wiper_slider)

        # Frame navigation
        nav_group = QGroupBox("Frame Navigation", self)
        nav_layout = QVBoxLayout(nav_group)

        # Navigation buttons row 1: frame steps
        nav_row1 = QHBoxLayout()
        self.prev_1sec_btn = QPushButton("-1s", self)
        self.prev_10_btn = QPushButton("-10", self)
        self.prev_1_btn = QPushButton("-1", self)
        self.next_1_btn = QPushButton("+1", self)
        self.next_10_btn = QPushButton("+10", self)
        self.next_1sec_btn = QPushButton("+1s", self)

        nav_row1.addWidget(self.prev_1sec_btn)
        nav_row1.addWidget(self.prev_10_btn)
        nav_row1.addWidget(self.prev_1_btn)
        nav_row1.addWidget(self.next_1_btn)
        nav_row1.addWidget(self.next_10_btn)
        nav_row1.addWidget(self.next_1sec_btn)
        nav_layout.addLayout(nav_row1)

        # Navigation row 2: frame number and timecode
        nav_row2 = QHBoxLayout()
        nav_row2.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox(self)
        self.frame_spinbox.setRange(0, 999999)
        self.frame_spinbox.setValue(0)
        nav_row2.addWidget(self.frame_spinbox)

        nav_row2.addSpacing(20)
        nav_row2.addWidget(QLabel("Timecode:"))
        self.timecode_edit = QLineEdit("00:00:00:00", self)
        self.timecode_edit.setMaximumWidth(120)
        nav_row2.addWidget(self.timecode_edit)
        self.go_to_tc_btn = QPushButton("Go", self)
        nav_row2.addWidget(self.go_to_tc_btn)

        nav_row2.addStretch()
        nav_layout.addLayout(nav_row2)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.frame_slider.setRange(0, 1000)
        self.frame_slider.setValue(0)
        nav_layout.addWidget(self.frame_slider)

        main_layout.addWidget(nav_group)

    def _connect_signals(self):
        """Connect UI signals."""
        # Mode selection
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # Zoom
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        self.zoom_100_btn.clicked.connect(self._zoom_100)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn.clicked.connect(self._zoom_out)

        # Navigation
        self.prev_1_btn.clicked.connect(lambda: self._navigate(-1))
        self.next_1_btn.clicked.connect(lambda: self._navigate(1))
        self.prev_10_btn.clicked.connect(lambda: self._navigate(-10))
        self.next_10_btn.clicked.connect(lambda: self._navigate(10))
        self.prev_1sec_btn.clicked.connect(lambda: self._navigate_seconds(-1))
        self.next_1sec_btn.clicked.connect(lambda: self._navigate_seconds(1))
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        self.go_to_tc_btn.clicked.connect(self._go_to_timecode)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        # State observation
        self.app_state.preview_frame_changed.connect(self._refresh_display)
        self.app_state.sources_changed.connect(self._on_sources_changed)

    @Slot(int)
    def _on_mode_changed(self, index: int):
        """Handle display mode change."""
        self._current_mode = self.mode_combo.itemData(index)
        self.wiper_slider.setVisible(
            self._current_mode == PreviewMode.WIPER
        )
        self._refresh_display()

    @Slot()
    def _zoom_fit(self):
        """Set zoom to fit view."""
        self._zoom_level = 1.0
        self.zoom_label.setText("Fit")
        self._refresh_display()

    @Slot()
    def _zoom_100(self):
        """Set zoom to 100%."""
        self._zoom_level = 1.0
        self.zoom_label.setText("100%")
        self._refresh_display()

    @Slot()
    def _zoom_in(self):
        """Zoom in."""
        self._zoom_level = min(self._zoom_level * 1.25, 8.0)
        self.zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        self._refresh_display()

    @Slot()
    def _zoom_out(self):
        """Zoom out."""
        self._zoom_level = max(self._zoom_level / 1.25, 0.25)
        self.zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        self._refresh_display()

    def _navigate(self, delta: int):
        """Navigate by frame delta."""
        new_frame = max(0, self.app_state.current_frame + delta)
        self.app_state.set_current_frame(new_frame)

    def _navigate_seconds(self, seconds: int):
        """Navigate by seconds (uses source FPS)."""
        fps = 24.0  # Default
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            fps = self.app_state.hdr_source.fps
        delta = int(seconds * fps)
        self._navigate(delta)

    @Slot(int)
    def _on_frame_spinbox_changed(self, value: int):
        """Handle direct frame number entry."""
        self.app_state.set_current_frame(value)

    @Slot()
    def _go_to_timecode(self):
        """Navigate to timecode entry."""
        tc = self.timecode_edit.text().strip()
        frame = self._timecode_to_frame(tc)
        if frame >= 0:
            self.app_state.set_current_frame(frame)

    @Slot(int)
    def _on_slider_changed(self, value: int):
        """Handle frame slider change."""
        # Map slider (0-1000) to actual frame range
        total = self._get_total_frames()
        if total > 0:
            frame = int(value * total / 1000)
            self.app_state.set_current_frame(frame)

    @Slot()
    def _on_sources_changed(self):
        """Update display when sources change."""
        total = self._get_total_frames()
        self.frame_spinbox.setRange(0, max(0, total - 1))
        self._refresh_display()

    @Slot()
    def _refresh_display(self):
        """Refresh frame display."""
        frame = self.app_state.current_frame

        # Update frame displays
        hdr_name = ""
        om_name = ""
        if self.app_state.hdr_source:
            hdr_name = self.app_state.hdr_source.filename
        if self.app_state.om_source:
            om_name = self.app_state.om_source.filename

        self.hdr_display.set_frame_info(frame, hdr_name)

        # Apply offset for OM display
        offset = 0
        if self.app_state.sync_proposal:
            offset = self.app_state.sync_proposal.offset
        om_frame = frame + offset
        self.om_display.set_frame_info(om_frame, om_name)

        # Update spinbox without triggering signal
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame)
        self.frame_spinbox.blockSignals(False)

        # Update timecode display
        self.timecode_edit.setText(self._frame_to_timecode(frame))

    def _get_total_frames(self) -> int:
        """Get total frame count from HDR source."""
        if self.app_state.hdr_source:
            return self.app_state.hdr_source.frame_count
        return 0

    def _frame_to_timecode(self, frame: int) -> str:
        """Convert frame number to timecode string."""
        fps = 24.0
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            fps = self.app_state.hdr_source.fps

        total_seconds = frame / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int(frame % fps)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _timecode_to_frame(self, tc: str) -> int:
        """Convert timecode string to frame number."""
        fps = 24.0
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            fps = self.app_state.hdr_source.fps

        try:
            parts = tc.split(":")
            if len(parts) == 4:
                h, m, s, f = [int(p) for p in parts]
                total_seconds = h * 3600 + m * 60 + s
                return int(total_seconds * fps) + f
        except (ValueError, IndexError):
            pass
        return -1
