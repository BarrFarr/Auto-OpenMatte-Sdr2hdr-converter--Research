"""
Quality Panel - Diagnostic metrics display.

Shows per-frame or per-shot quality diagnostics:
- Seam chroma
- Hue shift
- Delta E ITP (if available)
- Nonfinite pixel count
- Out-of-range pixel count

No global PASS/FAIL based on a single metric.
Metrics are informational for the operator.
"""
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QGridLayout,
    QFrame,
    QPushButton,
)
from PySide6.QtCore import Qt, Slot

from gui.models.state import AppState, QualityMetrics


class MetricWidget(QWidget):
    """Individual metric display with value and optional indicator."""

    def __init__(self, name: str, unit: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self.name_label = QLabel(f"{name}:", self)
        self.name_label.setProperty("class", "metric-name")
        self.name_label.setMinimumWidth(120)
        layout.addWidget(self.name_label)

        self.value_label = QLabel("-", self)
        self.value_label.setProperty("class", "metric-value")
        layout.addWidget(self.value_label)

        if unit:
            unit_label = QLabel(unit, self)
            unit_label.setProperty("class", "metric-unit")
            layout.addWidget(unit_label)

        layout.addStretch()

    def set_value(self, value, format_str: str = "{:.4f}"):
        """Update the displayed value."""
        if value is None:
            self.value_label.setText("N/A")
            self.value_label.setStyleSheet("color: #888888;")
        else:
            self.value_label.setText(format_str.format(value))
            self.value_label.setStyleSheet("")

    def set_int_value(self, value: int):
        """Update with integer value."""
        self.value_label.setText(str(value))
        if value > 0:
            self.value_label.setStyleSheet("color: #ffaa00;")
        else:
            self.value_label.setStyleSheet("color: #44cc44;")


class QualityPanel(QWidget):
    """Panel displaying diagnostic quality metrics."""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Info header
        info_label = QLabel(
            "Diagnostic quality metrics for the current frame/shot.\n"
            "These are informational - no automatic global PASS/FAIL.",
            self,
        )
        info_label.setWordWrap(True)
        info_label.setProperty("class", "info-text")
        main_layout.addWidget(info_label)

        # Metrics group
        metrics_group = QGroupBox("Quality Metrics", self)
        metrics_layout = QVBoxLayout(metrics_group)

        # Seam chroma
        self.seam_chroma_metric = MetricWidget("Seam Chroma", parent=self)
        metrics_layout.addWidget(self.seam_chroma_metric)

        # Hue shift
        self.hue_metric = MetricWidget("Hue Shift", unit="deg", parent=self)
        metrics_layout.addWidget(self.hue_metric)

        # Delta E ITP
        self.delta_e_metric = MetricWidget(
            "\u0394E ITP", parent=self
        )
        metrics_layout.addWidget(self.delta_e_metric)

        # Separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        metrics_layout.addWidget(sep)

        # Integrity checks
        self.nonfinite_metric = MetricWidget(
            "Nonfinite Pixels", parent=self
        )
        metrics_layout.addWidget(self.nonfinite_metric)

        self.oor_metric = MetricWidget(
            "Out-of-Range", parent=self
        )
        metrics_layout.addWidget(self.oor_metric)

        main_layout.addWidget(metrics_group)

        # Frame context
        frame_group = QGroupBox("Context", self)
        frame_layout = QHBoxLayout(frame_group)
        self.frame_label = QLabel("Frame: -", self)
        frame_layout.addWidget(self.frame_label)
        frame_layout.addStretch()
        self.refresh_btn = QPushButton("Refresh Metrics", self)
        frame_layout.addWidget(self.refresh_btn)
        main_layout.addWidget(frame_group)

        main_layout.addStretch()

    def _connect_signals(self):
        """Connect signals."""
        self.app_state.quality_changed.connect(self._refresh_display)
        self.refresh_btn.clicked.connect(self._request_refresh)

    @Slot()
    def _refresh_display(self):
        """Update display from current quality metrics."""
        metrics = self.app_state.quality_metrics
        if not metrics:
            self._clear_display()
            return

        self.frame_label.setText(f"Frame: {metrics.frame_index}")
        self.seam_chroma_metric.set_value(metrics.seam_chroma)
        self.hue_metric.set_value(metrics.hue_shift)
        self.delta_e_metric.set_value(metrics.delta_e_itp)
        self.nonfinite_metric.set_int_value(metrics.nonfinite_count)
        self.oor_metric.set_int_value(metrics.out_of_range_count)

    def _clear_display(self):
        """Clear all metric displays."""
        self.frame_label.setText("Frame: -")
        self.seam_chroma_metric.set_value(None)
        self.hue_metric.set_value(None)
        self.delta_e_metric.set_value(None)
        self.nonfinite_metric.set_int_value(0)
        self.oor_metric.set_int_value(0)

    @Slot()
    def _request_refresh(self):
        """Request quality metric refresh for current frame."""
        self.app_state.status_message.emit("Refreshing quality metrics...")
