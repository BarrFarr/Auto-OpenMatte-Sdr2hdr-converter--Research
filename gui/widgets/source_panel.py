"""
Source Panel - HDR and OpenMatte file selection.

Features:
- File selection via browse dialog
- Drag-and-drop support for video files
- ffprobe integration (calls backend inspect_source)
- File info display (resolution, fps, codec, duration, color space)
- Compatibility validation between HDR and OM sources
"""
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.controllers.pipeline_adapter import PipelineAdapter
from gui.models.state import AppState, SourceFileInfo


class FileDropWidget(QWidget):
    """Widget that accepts file drag-and-drop."""

    file_dropped = Signal(str)

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setProperty("class", "drop-zone")

        layout = QVBoxLayout(self)
        self.label = QLabel(label_text, self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.path_label = QLabel("Drop file here or click Browse", self)
        self.path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_label.setProperty("class", "hint-text")
        layout.addWidget(self.path_label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag if it contains file URLs."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop."""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)
            event.acceptProposedAction()

    def set_path(self, path: str):
        """Update the displayed file path."""
        if path:
            # Show only filename, full path in tooltip
            from pathlib import Path

            self.path_label.setText(Path(path).name)
            self.path_label.setToolTip(path)
        else:
            self.path_label.setText("Drop file here or click Browse")
            self.path_label.setToolTip("")


class FileInfoDisplay(QWidget):
    """Displays ffprobe-derived file information in a grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Info labels
        self._labels = {}
        fields = [
            ("Resolution", "resolution"),
            ("FPS", "fps"),
            ("Codec", "codec"),
            ("Pixel Format", "pix_fmt"),
            ("Duration", "duration"),
            ("Frames", "frame_count"),
            ("Color Space", "color_space"),
            ("Transfer", "color_transfer"),
            ("Primaries", "color_primaries"),
        ]

        for row, (display_name, key) in enumerate(fields):
            name_label = QLabel(f"{display_name}:", self)
            name_label.setProperty("class", "field-name")
            value_label = QLabel("-", self)
            value_label.setProperty("class", "field-value")
            layout.addWidget(name_label, row, 0)
            layout.addWidget(value_label, row, 1)
            self._labels[key] = value_label

    def update_info(self, info: SourceFileInfo):
        """Populate display with source file info."""
        if not info or not info.is_valid:
            self.clear()
            return

        self._labels["resolution"].setText(f"{info.width}x{info.height}")
        self._labels["fps"].setText(f"{info.fps:.3f}")
        self._labels["codec"].setText(info.codec)
        self._labels["pix_fmt"].setText(info.pix_fmt)

        # Format duration
        mins = int(info.duration_seconds // 60)
        secs = info.duration_seconds % 60
        self._labels["duration"].setText(f"{mins:02d}:{secs:05.2f}")

        self._labels["frame_count"].setText(str(info.frame_count))
        self._labels["color_space"].setText(info.color_space or "-")
        self._labels["color_transfer"].setText(info.color_transfer or "-")
        self._labels["color_primaries"].setText(info.color_primaries or "-")

    def clear(self):
        """Clear all displayed values."""
        for label in self._labels.values():
            label.setText("-")


class SourcePanel(QWidget):
    """Panel for selecting and validating HDR and OpenMatte source files."""

    def __init__(
        self, app_state: AppState, adapter: PipelineAdapter, parent=None
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.adapter = adapter

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # HDR Source group
        hdr_group = QGroupBox("HDR Source", self)
        hdr_layout = QVBoxLayout(hdr_group)

        # Drop zone and browse
        hdr_top = QHBoxLayout()
        self.hdr_drop = FileDropWidget("HDR Source", self)
        hdr_top.addWidget(self.hdr_drop, 1)

        hdr_buttons = QVBoxLayout()
        self.hdr_browse_btn = QPushButton("Browse...", self)
        self.hdr_clear_btn = QPushButton("Clear", self)
        hdr_buttons.addWidget(self.hdr_browse_btn)
        hdr_buttons.addWidget(self.hdr_clear_btn)
        hdr_buttons.addStretch()
        hdr_top.addLayout(hdr_buttons)

        hdr_layout.addLayout(hdr_top)

        # HDR info display
        self.hdr_info = FileInfoDisplay(self)
        hdr_layout.addWidget(self.hdr_info)
        main_layout.addWidget(hdr_group)

        # OM Source group
        om_group = QGroupBox("OpenMatte Source", self)
        om_layout = QVBoxLayout(om_group)

        # Drop zone and browse
        om_top = QHBoxLayout()
        self.om_drop = FileDropWidget("OpenMatte Source", self)
        om_top.addWidget(self.om_drop, 1)

        om_buttons = QVBoxLayout()
        self.om_browse_btn = QPushButton("Browse...", self)
        self.om_clear_btn = QPushButton("Clear", self)
        om_buttons.addWidget(self.om_browse_btn)
        om_buttons.addWidget(self.om_clear_btn)
        om_buttons.addStretch()
        om_top.addLayout(om_buttons)

        om_layout.addLayout(om_top)

        # OM info display
        self.om_info = FileInfoDisplay(self)
        om_layout.addWidget(self.om_info)
        main_layout.addWidget(om_group)

        # Validation section
        validation_group = QGroupBox("Compatibility Check", self)
        validation_layout = QVBoxLayout(validation_group)
        self.validation_label = QLabel("Select both sources to validate", self)
        self.validation_label.setProperty("class", "validation-info")
        validation_layout.addWidget(self.validation_label)
        main_layout.addWidget(validation_group)

        # Backend Status section
        backend_group = QGroupBox("Backend Status", self)
        backend_layout = QGridLayout(backend_group)
        backend_layout.setContentsMargins(4, 4, 4, 4)

        self._backend_labels = {}
        backend_fields = [
            ("Backend", "backend_name"),
            ("HW Decode", "hardware_decode"),
            ("HW Encode", "hardware_encode"),
            ("Zero-Copy", "zero_copy"),
            ("Bridge DLL", "bridge_status"),
        ]
        for row, (display_name, key) in enumerate(backend_fields):
            name_label = QLabel(f"{display_name}:", self)
            name_label.setProperty("class", "field-name")
            value_label = QLabel("-", self)
            value_label.setProperty("class", "field-value")
            backend_layout.addWidget(name_label, row, 0)
            backend_layout.addWidget(value_label, row, 1)
            self._backend_labels[key] = value_label

        main_layout.addWidget(backend_group)
        self._refresh_backend_status()

        main_layout.addStretch()

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        self.hdr_browse_btn.clicked.connect(self._browse_hdr)
        self.om_browse_btn.clicked.connect(self._browse_om)
        self.hdr_clear_btn.clicked.connect(self._clear_hdr)
        self.om_clear_btn.clicked.connect(self._clear_om)
        self.hdr_drop.file_dropped.connect(self._set_hdr_file)
        self.om_drop.file_dropped.connect(self._set_om_file)

        # State observation
        self.app_state.sources_changed.connect(self._refresh_display)

    @Slot()
    def _browse_hdr(self):
        """Open file dialog for HDR source."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDR Source",
            "",
            "Video Files (*.mkv *.mp4 *.mov *.avi *.mxf);;All Files (*)",
        )
        if path:
            self._set_hdr_file(path)

    @Slot()
    def _browse_om(self):
        """Open file dialog for OpenMatte source."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OpenMatte Source",
            "",
            "Video Files (*.mkv *.mp4 *.mov *.avi *.mxf);;All Files (*)",
        )
        if path:
            self._set_om_file(path)

    @Slot(str)
    def _set_hdr_file(self, path: str):
        """Set HDR source file and inspect it."""
        self.app_state.status_message.emit(f"Inspecting HDR: {path}")
        info = self.adapter.inspect_file(path)
        if info:
            self.hdr_drop.set_path(path)
            self.app_state.set_hdr_source(info)
        else:
            QMessageBox.warning(
                self, "Error", f"Failed to inspect HDR source:\n{path}"
            )

    @Slot(str)
    def _set_om_file(self, path: str):
        """Set OpenMatte source file and inspect it."""
        self.app_state.status_message.emit(f"Inspecting OM: {path}")
        info = self.adapter.inspect_file(path)
        if info:
            self.om_drop.set_path(path)
            self.app_state.set_om_source(info)
        else:
            QMessageBox.warning(
                self, "Error", f"Failed to inspect OpenMatte source:\n{path}"
            )

    @Slot()
    def _clear_hdr(self):
        """Clear HDR source."""
        self.hdr_drop.set_path("")
        self.hdr_info.clear()
        self.app_state.hdr_source = None
        self.app_state.sources_changed.emit()
        self._validate_compatibility()

    @Slot()
    def _clear_om(self):
        """Clear OpenMatte source."""
        self.om_drop.set_path("")
        self.om_info.clear()
        self.app_state.om_source = None
        self.app_state.sources_changed.emit()
        self._validate_compatibility()

    @Slot()
    def _refresh_display(self):
        """Refresh source paths and metadata from current state."""
        hdr = self.app_state.hdr_source
        if hdr:
            self.hdr_drop.set_path(hdr.path)
            if hdr.path and hdr.width <= 0:
                inspected = self.adapter.inspect_file(hdr.path)
                if inspected:
                    self.app_state.hdr_source = hdr = inspected
            self.hdr_info.update_info(hdr)
        else:
            self.hdr_drop.set_path("")
            self.hdr_info.clear()

        om = self.app_state.om_source
        if om:
            self.om_drop.set_path(om.path)
            if om.path and om.width <= 0:
                inspected = self.adapter.inspect_file(om.path)
                if inspected:
                    self.app_state.om_source = om = inspected
            self.om_info.update_info(om)
        else:
            self.om_drop.set_path("")
            self.om_info.clear()

        self._validate_compatibility()

    def _validate_compatibility(self):
        """Check compatibility between HDR and OM sources."""
        hdr = self.app_state.hdr_source
        om = self.app_state.om_source

        if not hdr or not om:
            self.validation_label.setText("Select both sources to validate")
            self.validation_label.setStyleSheet("")
            return

        issues = []

        # Check FPS match
        if hdr.fps > 0 and om.fps > 0:
            if abs(hdr.fps - om.fps) > 0.01:
                issues.append(
                    f"FPS mismatch: HDR={hdr.fps:.3f}, OM={om.fps:.3f}"
                )

        # Check resolution (OM should be >= HDR or same aspect)
        if hdr.width > 0 and om.width > 0:
            hdr_aspect = hdr.width / max(hdr.height, 1)
            om_aspect = om.width / max(om.height, 1)
            if abs(hdr_aspect - om_aspect) > 0.02:
                issues.append(
                    f"Aspect ratio mismatch: "
                    f"HDR={hdr.width}x{hdr.height}, "
                    f"OM={om.width}x{om.height}"
                )

        if issues:
            self.validation_label.setText(
                "WARNINGS:\n" + "\n".join(f"  - {i}" for i in issues)
            )
            self.validation_label.setStyleSheet("color: #ffaa00;")
        else:
            self.validation_label.setText("Sources compatible")
            self.validation_label.setStyleSheet("color: #44cc44;")

    def _refresh_backend_status(self):
        """Query backend status and update the Backend Status group display."""
        try:
            status = self.adapter.get_backend_status()
        except Exception:
            status = {
                "available": False,
                "backend_name": "",
                "hardware_decode": False,
                "hardware_encode": False,
                "zero_copy": False,
                "supported_codecs": [],
            }

        if status.get("available"):
            self._backend_labels["backend_name"].setText(
                status.get("backend_name", "Unknown")
            )
            self._backend_labels["backend_name"].setStyleSheet("color: #44cc44;")
        elif status.get("backend_name"):
            self._backend_labels["backend_name"].setText(
                f"{status['backend_name']} (not active)"
            )
            self._backend_labels["backend_name"].setStyleSheet("color: #ffaa00;")
        else:
            self._backend_labels["backend_name"].setText("Not available")
            self._backend_labels["backend_name"].setStyleSheet("color: #888888;")

        def _bool_label(value: bool) -> str:
            return "Yes" if value else "No"

        self._backend_labels["hardware_decode"].setText(
            _bool_label(status.get("hardware_decode", False))
        )
        self._backend_labels["hardware_encode"].setText(
            _bool_label(status.get("hardware_encode", False))
        )
        self._backend_labels["zero_copy"].setText(
            _bool_label(status.get("zero_copy", False))
        )

        # Bridge DLL info
        try:
            bridge_info = self.adapter.get_bridge_info()
        except Exception:
            bridge_info = {"bridge_exists": False, "manifest_found": False}

        if bridge_info.get("bridge_exists"):
            self._backend_labels["bridge_status"].setText("Present")
            self._backend_labels["bridge_status"].setStyleSheet("color: #44cc44;")
        elif bridge_info.get("manifest_found"):
            self._backend_labels["bridge_status"].setText("Manifest found, DLL missing")
            self._backend_labels["bridge_status"].setStyleSheet("color: #ffaa00;")
        else:
            self._backend_labels["bridge_status"].setText("Not found")
            self._backend_labels["bridge_status"].setStyleSheet("color: #888888;")
