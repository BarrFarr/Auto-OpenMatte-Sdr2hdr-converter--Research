"""
Output Preview Panel - View processed output with comparison modes.

Provides three viewing modes:
- Current: Shows the current processed output frame
- Before/After: Split view comparing source vs processed
- A/B Toggle: Switch between two views with a button
"""
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
    QStackedWidget,
)
from PySide6.QtCore import Qt, Slot

from gui.models.state import AppState


class OutputViewMode:
    """Constants for output preview modes."""

    CURRENT = "current"
    BEFORE_AFTER = "before_after"
    AB_TOGGLE = "ab_toggle"


class OutputPreviewPanel(QWidget):
    """Panel for viewing processed output with comparison modes."""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._current_mode = OutputViewMode.CURRENT
        self._showing_a = True  # For A/B toggle

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Mode selection toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("View Mode:"))

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("Current", OutputViewMode.CURRENT)
        self.mode_combo.addItem("Before / After", OutputViewMode.BEFORE_AFTER)
        self.mode_combo.addItem("A/B Toggle", OutputViewMode.AB_TOGGLE)
        toolbar.addWidget(self.mode_combo)

        # A/B toggle button (only visible in A/B mode)
        self.ab_toggle_btn = QPushButton("Show B", self)
        self.ab_toggle_btn.setVisible(False)
        self.ab_toggle_btn.setProperty("class", "toggle-button")
        toolbar.addWidget(self.ab_toggle_btn)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Display area
        display_frame = QFrame(self)
        display_frame.setFrameShape(QFrame.Shape.StyledPanel)
        display_layout = QVBoxLayout(display_frame)

        # Stacked widget for different view modes
        self.view_stack = QStackedWidget(self)

        # Current view (single output frame)
        self.current_view = QLabel(
            "[Output Preview]\nProcessed frame will appear here", self
        )
        self.current_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_view.setMinimumSize(640, 360)
        self.current_view.setProperty("class", "frame-display")
        self.view_stack.addWidget(self.current_view)

        # Before/After view (side by side)
        before_after_widget = QWidget(self)
        ba_layout = QHBoxLayout(before_after_widget)
        self.before_label = QLabel(
            "[Before]\nSource frame", self
        )
        self.before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.before_label.setProperty("class", "frame-display")
        self.after_label = QLabel(
            "[After]\nProcessed frame", self
        )
        self.after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.after_label.setProperty("class", "frame-display")
        ba_layout.addWidget(self.before_label)
        ba_layout.addWidget(self.after_label)
        self.view_stack.addWidget(before_after_widget)

        # A/B toggle view (single frame, toggled)
        self.ab_view = QLabel(
            "[A]\nToggle between views", self
        )
        self.ab_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ab_view.setMinimumSize(640, 360)
        self.ab_view.setProperty("class", "frame-display")
        self.view_stack.addWidget(self.ab_view)

        display_layout.addWidget(self.view_stack)
        main_layout.addWidget(display_frame, 1)

        # Frame info
        info_layout = QHBoxLayout()
        self.frame_info_label = QLabel("Frame: -", self)
        self.output_info_label = QLabel("Output: Not rendered", self)
        info_layout.addWidget(self.frame_info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.output_info_label)
        main_layout.addLayout(info_layout)

    def _connect_signals(self):
        """Connect signals."""
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.ab_toggle_btn.clicked.connect(self._toggle_ab)
        self.app_state.preview_frame_changed.connect(self._refresh_display)

    @Slot(int)
    def _on_mode_changed(self, index: int):
        """Handle view mode change."""
        self._current_mode = self.mode_combo.itemData(index)

        if self._current_mode == OutputViewMode.CURRENT:
            self.view_stack.setCurrentIndex(0)
            self.ab_toggle_btn.setVisible(False)
        elif self._current_mode == OutputViewMode.BEFORE_AFTER:
            self.view_stack.setCurrentIndex(1)
            self.ab_toggle_btn.setVisible(False)
        elif self._current_mode == OutputViewMode.AB_TOGGLE:
            self.view_stack.setCurrentIndex(2)
            self.ab_toggle_btn.setVisible(True)

    @Slot()
    def _toggle_ab(self):
        """Toggle between A and B views."""
        self._showing_a = not self._showing_a
        if self._showing_a:
            self.ab_toggle_btn.setText("Show B")
            self.ab_view.setText("[A]\nSource / Reference view")
        else:
            self.ab_toggle_btn.setText("Show A")
            self.ab_view.setText("[B]\nProcessed / Output view")

    @Slot()
    def _refresh_display(self):
        """Refresh display with current frame info."""
        frame = self.app_state.current_frame
        self.frame_info_label.setText(f"Frame: {frame}")

        # Update display labels
        self.current_view.setText(
            f"[Output Preview]\nFrame: {frame}\n"
            "(Rendered output will appear here)"
        )
        self.before_label.setText(f"[Before]\nFrame: {frame}")
        self.after_label.setText(f"[After]\nFrame: {frame}")
