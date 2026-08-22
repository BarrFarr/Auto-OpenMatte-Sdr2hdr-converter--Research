"""
Render Panel - Output configuration and render progress monitoring.

Provides:
- Output path selection
- Codec / quality (CRF) / pixel format configuration
- Resolution setting
- Render progress: progress bar, FPS, ETA, elapsed, VRAM usage
- Start/Stop render controls
"""
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QProgressBar,
    QGridLayout,
    QFileDialog,
)
from PySide6.QtCore import Qt, Slot, QTimer

from gui.models.state import AppState, RenderConfig, RenderStatus
from gui.controllers.render_controller import RenderController


class RenderPanel(QWidget):
    """Panel for output configuration and render monitoring."""

    def __init__(
        self,
        app_state: AppState,
        render_controller: RenderController,
        parent=None,
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.render_controller = render_controller

        self._setup_ui()
        self._connect_signals()
        self._load_config_to_ui()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Output configuration
        config_group = QGroupBox("Output Configuration", self)
        config_layout = QGridLayout(config_group)

        # Output path
        config_layout.addWidget(QLabel("Output Path:"), 0, 0)
        self.output_path_edit = QLineEdit(self)
        self.output_path_edit.setPlaceholderText(
            "Select output file location..."
        )
        config_layout.addWidget(self.output_path_edit, 0, 1)
        self.browse_output_btn = QPushButton("Browse...", self)
        config_layout.addWidget(self.browse_output_btn, 0, 2)

        # Codec
        config_layout.addWidget(QLabel("Codec:"), 1, 0)
        self.codec_combo = QComboBox(self)
        self.codec_combo.addItems([
            "libx265",
            "libx264",
            "hevc_nvenc",
            "h264_nvenc",
        ])
        config_layout.addWidget(self.codec_combo, 1, 1)

        # CRF / Quality
        config_layout.addWidget(QLabel("CRF / Quality:"), 2, 0)
        self.crf_spinbox = QSpinBox(self)
        self.crf_spinbox.setRange(0, 51)
        self.crf_spinbox.setValue(16)
        self.crf_spinbox.setToolTip("Lower = higher quality. 0 = lossless.")
        config_layout.addWidget(self.crf_spinbox, 2, 1)

        # Pixel format
        config_layout.addWidget(QLabel("Pixel Format:"), 3, 0)
        self.pix_fmt_combo = QComboBox(self)
        self.pix_fmt_combo.addItems([
            "yuv420p10le",
            "yuv420p",
            "yuv444p10le",
            "yuv444p",
        ])
        config_layout.addWidget(self.pix_fmt_combo, 3, 1)

        # Resolution
        config_layout.addWidget(QLabel("Resolution:"), 4, 0)
        self.resolution_edit = QLineEdit(self)
        self.resolution_edit.setPlaceholderText(
            "Empty = source resolution (e.g. 3840x2160)"
        )
        config_layout.addWidget(self.resolution_edit, 4, 1)

        main_layout.addWidget(config_group)

        # Render controls
        control_group = QGroupBox("Render Control", self)
        control_layout = QHBoxLayout(control_group)

        self.start_render_btn = QPushButton("Start Render", self)
        self.start_render_btn.setProperty("class", "primary-button")
        self.start_render_btn.setMinimumHeight(36)
        control_layout.addWidget(self.start_render_btn)

        self.stop_render_btn = QPushButton("Stop", self)
        self.stop_render_btn.setEnabled(False)
        control_layout.addWidget(self.stop_render_btn)

        control_layout.addStretch()
        main_layout.addWidget(control_group)

        # Progress section
        progress_group = QGroupBox("Render Progress", self)
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        # Stats grid
        stats_layout = QGridLayout()

        stats_layout.addWidget(QLabel("Frames:"), 0, 0)
        self.frames_label = QLabel("0 / 0", self)
        stats_layout.addWidget(self.frames_label, 0, 1)

        stats_layout.addWidget(QLabel("FPS:"), 0, 2)
        self.fps_label = QLabel("-", self)
        stats_layout.addWidget(self.fps_label, 0, 3)

        stats_layout.addWidget(QLabel("ETA:"), 1, 0)
        self.eta_label = QLabel("-", self)
        stats_layout.addWidget(self.eta_label, 1, 1)

        stats_layout.addWidget(QLabel("Elapsed:"), 1, 2)
        self.elapsed_label = QLabel("-", self)
        stats_layout.addWidget(self.elapsed_label, 1, 3)

        stats_layout.addWidget(QLabel("VRAM:"), 2, 0)
        self.vram_label = QLabel("-", self)
        stats_layout.addWidget(self.vram_label, 2, 1)

        stats_layout.addWidget(QLabel("Output:"), 2, 2)
        self.output_status_label = QLabel("-", self)
        stats_layout.addWidget(self.output_status_label, 2, 3)

        progress_layout.addLayout(stats_layout)

        # Error display
        self.error_label = QLabel("", self)
        self.error_label.setStyleSheet("color: #ff4444;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        progress_layout.addWidget(self.error_label)

        main_layout.addWidget(progress_group)
        main_layout.addStretch()

    def _connect_signals(self):
        """Connect signals."""
        self.browse_output_btn.clicked.connect(self._browse_output)
        self.start_render_btn.clicked.connect(self._start_render)
        self.stop_render_btn.clicked.connect(self._stop_render)

        # Config changes
        self.codec_combo.currentTextChanged.connect(self._update_config)
        self.crf_spinbox.valueChanged.connect(self._update_config)
        self.pix_fmt_combo.currentTextChanged.connect(self._update_config)
        self.resolution_edit.textChanged.connect(self._update_config)
        self.output_path_edit.textChanged.connect(self._update_config)

        # State observation
        self.app_state.render_status_changed.connect(self._refresh_progress)

    def _load_config_to_ui(self):
        """Load current render config into UI elements."""
        rc = self.app_state.render_config
        self.output_path_edit.setText(rc.output_path)
        index = self.codec_combo.findText(rc.codec)
        if index >= 0:
            self.codec_combo.setCurrentIndex(index)
        self.crf_spinbox.setValue(rc.crf)
        pix_index = self.pix_fmt_combo.findText(rc.pix_fmt)
        if pix_index >= 0:
            self.pix_fmt_combo.setCurrentIndex(pix_index)
        self.resolution_edit.setText(rc.resolution)

    @Slot()
    def _browse_output(self):
        """Open file dialog for output path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "Video Files (*.mkv *.mp4 *.mov);;All Files (*)",
        )
        if path:
            self.output_path_edit.setText(path)

    @Slot()
    def _update_config(self):
        """Update render config from UI."""
        self.app_state.render_config = RenderConfig(
            output_path=self.output_path_edit.text(),
            codec=self.codec_combo.currentText(),
            crf=self.crf_spinbox.value(),
            pix_fmt=self.pix_fmt_combo.currentText(),
            resolution=self.resolution_edit.text(),
        )
        self.app_state.is_dirty = True

    @Slot()
    def _start_render(self):
        """Start the render process."""
        self.start_render_btn.setEnabled(False)
        self.stop_render_btn.setEnabled(True)
        self.error_label.setVisible(False)
        self.render_controller.start_render()

    @Slot()
    def _stop_render(self):
        """Stop the render process."""
        self.render_controller.stop_render()
        self.start_render_btn.setEnabled(True)
        self.stop_render_btn.setEnabled(False)

    @Slot()
    def _refresh_progress(self):
        """Update progress display from render status."""
        status = self.app_state.render_status

        # Progress bar
        self.progress_bar.setValue(int(status.progress_percent))

        # Frames
        self.frames_label.setText(
            f"{status.current_frame} / {status.total_frames}"
        )

        # FPS
        if status.fps > 0:
            self.fps_label.setText(f"{status.fps:.1f}")
        else:
            self.fps_label.setText("-")

        # ETA
        if status.eta_seconds > 0:
            self.eta_label.setText(self._format_time(status.eta_seconds))
        else:
            self.eta_label.setText("-")

        # Elapsed
        if status.elapsed_seconds > 0:
            self.elapsed_label.setText(
                self._format_time(status.elapsed_seconds)
            )
        else:
            self.elapsed_label.setText("-")

        # VRAM
        if status.vram_usage_mb > 0:
            self.vram_label.setText(f"{status.vram_usage_mb:.0f} MB")
        else:
            self.vram_label.setText("-")

        # Output path
        if status.output_path:
            self.output_status_label.setText(status.output_path)

        # Error
        if status.error_message:
            self.error_label.setText(status.error_message)
            self.error_label.setVisible(True)

        # Button states
        if not status.is_rendering:
            self.start_render_btn.setEnabled(True)
            self.stop_render_btn.setEnabled(False)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
