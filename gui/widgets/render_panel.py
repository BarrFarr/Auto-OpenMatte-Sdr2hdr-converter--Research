"""Render configuration and segmented render lifecycle controls."""

from __future__ import annotations

from pathlib import Path

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
from PySide6.QtCore import Slot

from gui.models.state import AppState, RenderConfig, RenderStatus
from gui.controllers.render_controller import RenderController


class RenderPanel(QWidget):
    """Panel for output settings, audio selection and resumable rendering."""

    def __init__(self, app_state: AppState, render_controller: RenderController, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.render_controller = render_controller
        self._updating_audio = False
        self._setup_ui()
        self._connect_signals()
        self._load_config_to_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        config_group = QGroupBox("Output Configuration", self)
        config_layout = QGridLayout(config_group)
        config_layout.addWidget(QLabel("Output Path:"), 0, 0)
        self.output_path_edit = QLineEdit(self)
        self.output_path_edit.setPlaceholderText("Select output file location...")
        config_layout.addWidget(self.output_path_edit, 0, 1)
        self.browse_output_btn = QPushButton("Browse...", self)
        config_layout.addWidget(self.browse_output_btn, 0, 2)

        config_layout.addWidget(QLabel("Codec:"), 1, 0)
        self.codec_combo = QComboBox(self)
        self.codec_combo.addItems(["libx265", "libx264", "hevc_nvenc", "h264_nvenc"])
        config_layout.addWidget(self.codec_combo, 1, 1)

        config_layout.addWidget(QLabel("CRF / Quality:"), 2, 0)
        self.crf_spinbox = QSpinBox(self)
        self.crf_spinbox.setRange(0, 51)
        self.crf_spinbox.setValue(16)
        config_layout.addWidget(self.crf_spinbox, 2, 1)

        config_layout.addWidget(QLabel("Pixel Format:"), 3, 0)
        self.pix_fmt_combo = QComboBox(self)
        self.pix_fmt_combo.addItems(["yuv420p10le", "yuv420p", "yuv444p10le", "yuv444p"])
        config_layout.addWidget(self.pix_fmt_combo, 3, 1)

        config_layout.addWidget(QLabel("Resolution:"), 4, 0)
        self.resolution_edit = QLineEdit(self)
        self.resolution_edit.setPlaceholderText("Empty = source resolution (e.g. 3840x2160)")
        config_layout.addWidget(self.resolution_edit, 4, 1)

        config_layout.addWidget(QLabel("Segment frames:"), 5, 0)
        self.segment_frames_spinbox = QSpinBox(self)
        self.segment_frames_spinbox.setRange(1, 1_000_000)
        self.segment_frames_spinbox.setToolTip("Checkpoint boundary size; completed segments are never appended")
        config_layout.addWidget(self.segment_frames_spinbox, 5, 1)

        config_layout.addWidget(QLabel("Audio:"), 6, 0)
        self.audio_source_combo = QComboBox(self)
        self.audio_source_combo.addItem("NONE", "NONE")
        self.audio_source_combo.addItem("FROM HDR REFERENCE", "HDR")
        self.audio_source_combo.addItem("FROM OPEN MATTE", "OM")
        config_layout.addWidget(self.audio_source_combo, 6, 1)

        config_layout.addWidget(QLabel("Audio track:"), 7, 0)
        self.audio_track_combo = QComboBox(self)
        self.audio_track_combo.setPlaceholderText("No audio selected")
        config_layout.addWidget(self.audio_track_combo, 7, 1, 1, 2)
        main_layout.addWidget(config_group)

        control_group = QGroupBox("Render Control", self)
        control_layout = QHBoxLayout(control_group)
        self.start_render_btn = QPushButton("RENDER", self)
        self.start_render_btn.setProperty("class", "primary-button")
        self.start_render_btn.setMinimumHeight(36)
        control_layout.addWidget(self.start_render_btn)
        self.pause_render_btn = QPushButton("PAUSE", self)
        self.pause_render_btn.setEnabled(False)
        control_layout.addWidget(self.pause_render_btn)
        self.resume_render_btn = QPushButton("RESUME", self)
        self.resume_render_btn.setEnabled(False)
        control_layout.addWidget(self.resume_render_btn)
        self.stop_render_btn = QPushButton("CANCEL", self)
        self.stop_render_btn.setEnabled(False)
        control_layout.addWidget(self.stop_render_btn)
        control_layout.addStretch()
        main_layout.addWidget(control_group)

        progress_group = QGroupBox("Render Progress", self)
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        stats_layout = QGridLayout()
        stats_layout.addWidget(QLabel("State:"), 0, 0)
        self.state_label = QLabel("IDLE", self)
        stats_layout.addWidget(self.state_label, 0, 1)
        stats_layout.addWidget(QLabel("Frames:"), 0, 2)
        self.frames_label = QLabel("0 / 0", self)
        stats_layout.addWidget(self.frames_label, 0, 3)
        stats_layout.addWidget(QLabel("Segment:"), 1, 0)
        self.segment_label = QLabel("-", self)
        stats_layout.addWidget(self.segment_label, 1, 1)
        stats_layout.addWidget(QLabel("Last committed:"), 1, 2)
        self.committed_label = QLabel("-", self)
        stats_layout.addWidget(self.committed_label, 1, 3)
        stats_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_label = QLabel("-", self)
        stats_layout.addWidget(self.fps_label, 2, 1)
        stats_layout.addWidget(QLabel("ETA:"), 2, 2)
        self.eta_label = QLabel("-", self)
        stats_layout.addWidget(self.eta_label, 2, 3)
        stats_layout.addWidget(QLabel("Elapsed:"), 3, 0)
        self.elapsed_label = QLabel("-", self)
        stats_layout.addWidget(self.elapsed_label, 3, 1)
        stats_layout.addWidget(QLabel("VRAM:"), 3, 2)
        self.vram_label = QLabel("-", self)
        stats_layout.addWidget(self.vram_label, 3, 3)
        stats_layout.addWidget(QLabel("Output:"), 4, 0)
        self.output_status_label = QLabel("-", self)
        stats_layout.addWidget(self.output_status_label, 4, 1, 1, 3)
        progress_layout.addLayout(stats_layout)
        self.resume_label = QLabel("", self)
        self.resume_label.setStyleSheet("color: #ffaa00;")
        progress_layout.addWidget(self.resume_label)
        self.error_label = QLabel("", self)
        self.error_label.setStyleSheet("color: #ff4444;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        progress_layout.addWidget(self.error_label)
        main_layout.addWidget(progress_group)
        main_layout.addStretch()

    def _connect_signals(self):
        self.browse_output_btn.clicked.connect(self._browse_output)
        self.start_render_btn.clicked.connect(self._start_render)
        self.pause_render_btn.clicked.connect(self._pause_render)
        self.resume_render_btn.clicked.connect(self._resume_render)
        self.stop_render_btn.clicked.connect(self._stop_render)
        for widget_signal in (
            self.codec_combo.currentTextChanged,
            self.pix_fmt_combo.currentTextChanged,
            self.resolution_edit.textChanged,
        ):
            widget_signal.connect(self._update_config)
        self.crf_spinbox.valueChanged.connect(self._update_config)
        self.segment_frames_spinbox.valueChanged.connect(self._update_config)
        self.output_path_edit.textChanged.connect(self._update_config)
        self.audio_source_combo.currentIndexChanged.connect(self._audio_source_changed)
        self.audio_track_combo.currentIndexChanged.connect(self._audio_track_changed)
        self.app_state.render_status_changed.connect(self._refresh_progress)
        self.app_state.sources_changed.connect(self._refresh_audio_tracks)
        self.app_state.project_changed.connect(self._refresh_progress)

    def _load_config_to_ui(self):
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
        self.segment_frames_spinbox.setValue(max(1, int(rc.segment_frames)))
        source_index = self.audio_source_combo.findData(rc.audio_source)
        self.audio_source_combo.setCurrentIndex(source_index if source_index >= 0 else 0)
        self._refresh_audio_tracks()

    @Slot()
    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "Video Files (*.mkv *.mp4 *.mov);;All Files (*)")
        if path:
            self.output_path_edit.setText(path)

    def _selected_audio_track(self):
        data = self.audio_track_combo.currentData()
        return data if isinstance(data, dict) else None

    @Slot()
    def _update_config(self):
        if self._updating_audio:
            return
        old = self.app_state.render_config
        track = self._selected_audio_track()
        source = str(self.audio_source_combo.currentData() or "NONE")
        if source == "NONE":
            track = None
        self.app_state.render_config = RenderConfig(
            output_path=self.output_path_edit.text(),
            codec=self.codec_combo.currentText(),
            crf=self.crf_spinbox.value(),
            pix_fmt=self.pix_fmt_combo.currentText(),
            resolution=self.resolution_edit.text(),
            backend_project_path=old.backend_project_path,
            segment_frames=self.segment_frames_spinbox.value(),
            checkpoint_path=old.checkpoint_path,
            audio_source=source,
            audio_stream_ordinal=int(track.get("ordinal", -1)) if track else -1,
            audio_stream_index=int(track.get("global_index", -1)) if track else -1,
            audio_codec=str(track.get("codec_name", "")) if track else "",
            audio_language=str(track.get("language", "")) if track else "",
            audio_title=str(track.get("title", "")) if track else "",
        )
        self.app_state.is_dirty = True

    @Slot()
    def _audio_source_changed(self):
        self._refresh_audio_tracks()
        self._update_config()

    @Slot()
    def _audio_track_changed(self):
        self._update_config()

    @Slot()
    def _refresh_audio_tracks(self):
        if not hasattr(self, "audio_track_combo"):
            return
        source = str(self.audio_source_combo.currentData() or "NONE")
        info = self.app_state.hdr_source if source == "HDR" else self.app_state.om_source if source == "OM" else None
        self._updating_audio = True
        try:
            current_ordinal = self.app_state.render_config.audio_stream_ordinal
            self.audio_track_combo.clear()
            if info is None or not info.audio_tracks:
                self.audio_track_combo.addItem("No audio tracks found", None)
                return
            for track in info.audio_tracks:
                data = {
                    "ordinal": track.ordinal,
                    "global_index": track.global_index,
                    "codec_name": track.codec_name,
                    "language": track.language,
                    "title": track.title,
                }
                self.audio_track_combo.addItem(f"Track {track.ordinal + 1} — {track.label}", data)
            for index in range(self.audio_track_combo.count()):
                data = self.audio_track_combo.itemData(index)
                if isinstance(data, dict) and int(data.get("ordinal", -1)) == current_ordinal:
                    self.audio_track_combo.setCurrentIndex(index)
                    break
        finally:
            self._updating_audio = False

    @Slot()
    def _start_render(self):
        self.start_render_btn.setEnabled(False)
        self.pause_render_btn.setEnabled(True)
        self.resume_render_btn.setEnabled(False)
        self.stop_render_btn.setEnabled(True)
        self.error_label.setVisible(False)
        self.render_controller.start_render()

    @Slot()
    def _pause_render(self):
        self.pause_render_btn.setEnabled(False)
        self.render_controller.pause_render()

    @Slot()
    def _resume_render(self):
        self.resume_render_btn.setEnabled(False)
        self.render_controller.resume_render()

    @Slot()
    def _stop_render(self):
        self.stop_render_btn.setEnabled(False)
        self.render_controller.stop_render()

    @Slot()
    def _refresh_progress(self):
        status = self.app_state.render_status
        self.progress_bar.setValue(int(status.progress_percent))
        self.state_label.setText(status.state)
        self.frames_label.setText(f"{status.current_frame} / {status.total_frames}")
        self.segment_label.setText(
            "-" if status.current_segment is None else f"{status.current_segment + 1} / {status.total_segments}"
        )
        self.committed_label.setText(str(status.last_committed_frame) if status.last_committed_frame >= 0 else "-")
        self.fps_label.setText(f"{status.fps:.1f}" if status.fps > 0 else "-")
        self.eta_label.setText(self._format_time(status.eta_seconds) if status.eta_seconds > 0 else "-")
        self.elapsed_label.setText(self._format_time(status.elapsed_seconds) if status.elapsed_seconds > 0 else "-")
        self.vram_label.setText(f"{status.vram_usage_mb:.0f} MB" if status.vram_usage_mb > 0 else "-")
        if status.output_path:
            self.output_status_label.setText(status.output_path)
        if status.error_message:
            self.error_label.setText(status.error_message)
            self.error_label.setVisible(True)
        is_running = status.is_rendering
        self.start_render_btn.setEnabled(not is_running and status.state not in {"PAUSED", "CANCELLED"})
        self.pause_render_btn.setEnabled(is_running and status.state not in {"PAUSE_REQUESTED", "CANCEL_REQUESTED"})
        self.stop_render_btn.setEnabled(is_running)
        checkpoint_path = status.checkpoint_path
        if not checkpoint_path:
            checkpoint_path = self.render_controller._checkpoint_path()
        has_checkpoint = bool(
            checkpoint_path and Path(checkpoint_path).expanduser().is_file()
        )
        resumable = (status.state in {"PAUSED", "CANCELLED"} or has_checkpoint)
        self.resume_render_btn.setEnabled(
            not is_running and resumable and status.state != "COMPLETE"
        )
        if checkpoint_path and status.state == "PAUSED":
            self.resume_label.setText(
                f"Resume available: frame {status.last_committed_frame}, "
                f"checkpoint {checkpoint_path}"
            )
        elif has_checkpoint:
            self.resume_label.setText(f"Resume checkpoint found: {checkpoint_path}")
        else:
            self.resume_label.setText("")

    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
