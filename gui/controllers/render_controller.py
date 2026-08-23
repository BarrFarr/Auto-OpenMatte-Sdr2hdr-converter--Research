"""Render lifecycle controller for segmented, resumable jobs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from gui.models.state import AppState, RenderStatus
from gui.controllers.pipeline_adapter import PipelineAdapter


class RenderController(QObject):
    """Controls render lifecycle without owning image-processing logic."""

    render_started = Signal()
    render_stopped = Signal()
    render_completed = Signal(bool)

    def __init__(self, app_state: AppState, adapter: PipelineAdapter, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.adapter = adapter
        self._start_time: Optional[float] = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(500)
        self._progress_timer.timeout.connect(self._poll_progress)
        self.adapter.render_completed.connect(self._on_render_finished)
        self.adapter.render_progress.connect(self._on_progress_update)
        self.adapter.error_occurred.connect(self._on_error)

    def _checkpoint_path(self) -> str:
        rc = self.app_state.render_config
        if rc.checkpoint_path:
            return rc.checkpoint_path
        if self.app_state.project_path:
            return str(Path(self.app_state.project_path).resolve().with_name("render_state.json"))
        if rc.output_path:
            return str(Path(rc.output_path).resolve().with_name("render_state.json"))
        return ""

    def start_render(self, *, resume: bool = False):
        """Start a new segmented render or reconcile an existing checkpoint."""
        rc = self.app_state.render_config
        if not rc.output_path:
            self.app_state.status_message.emit("Cannot render: no output path specified")
            return
        if not self.app_state.hdr_source or not self.app_state.om_source:
            self.app_state.status_message.emit("Cannot render: both sources required")
            return
        locked = [sl for sl in self.app_state.shot_locks if sl.is_locked]
        if not locked:
            self.app_state.status_message.emit("Cannot render: no shots are locked")
            return

        status = RenderStatus(
            is_rendering=True,
            progress_percent=0.0,
            current_frame=0,
            total_frames=int(self.app_state.hdr_source.frame_count or 0),
            output_path=rc.output_path,
            processing_mode=str(self.app_state.processing_mode.value),
            state="RUNNING",
            checkpoint_path=self._checkpoint_path(),
        )
        self.app_state.set_render_status(status)
        self._start_time = time.time()
        self._progress_timer.start()
        self.render_started.emit()
        self.app_state.status_message.emit(
            ("Resuming" if resume else "Rendering") + f" to: {rc.output_path}"
        )
        self.adapter.start_render(resume=resume)

    def resume_render(self):
        checkpoint_value = self._checkpoint_path()
        checkpoint = Path(checkpoint_value) if checkpoint_value else None
        checkpoint_exists = bool(checkpoint and checkpoint.is_file())
        status = self.app_state.render_status
        if not checkpoint_exists:
            self.app_state.status_message.emit("No resumable render checkpoint found")
            return
        if not status.checkpoint_path and checkpoint is not None:
            status.checkpoint_path = str(checkpoint)
            self.app_state.set_render_status(status)
        self.start_render(resume=True)

    def pause_render(self):
        """Ask the worker to stop at the next safe segment boundary."""
        if not self.app_state.render_status.is_rendering:
            return
        self.adapter.pause_render()
        status = self.app_state.render_status
        status.state = "PAUSE_REQUESTED"
        self.app_state.set_render_status(status)
        self.app_state.status_message.emit("Pause requested; closing current segment safely")

    def stop_render(self):
        """Ask the worker to cancel at a safe frame boundary."""
        if not self.app_state.render_status.is_rendering:
            return
        self.adapter.stop_render()
        status = self.app_state.render_status
        status.state = "CANCEL_REQUESTED"
        self.app_state.set_render_status(status)
        self.app_state.status_message.emit("Cancel requested; closing current segment safely")

    @Slot()
    def _poll_progress(self):
        if not self._start_time:
            return
        elapsed = time.time() - self._start_time
        status = self.app_state.render_status
        status.elapsed_seconds = elapsed
        if status.progress_percent > 0:
            total_estimated = elapsed / (status.progress_percent / 100.0)
            status.eta_seconds = max(0.0, total_estimated - elapsed)
        if elapsed > 0 and status.current_frame > 0:
            status.fps = status.current_frame / elapsed
        status.vram_usage_mb = self._get_vram_usage()
        self.app_state.set_render_status(status)

    @Slot(object)
    def _on_progress_update(self, progress_data):
        if isinstance(progress_data, RenderStatus):
            current = self.app_state.render_status
            progress_data.elapsed_seconds = current.elapsed_seconds
            progress_data.fps = current.fps
            progress_data.eta_seconds = current.eta_seconds
            progress_data.vram_usage_mb = current.vram_usage_mb
            self.app_state.set_render_status(progress_data)

    @Slot(bool)
    def _on_render_finished(self, success: bool):
        self._progress_timer.stop()
        status = self.app_state.render_status
        if status.state == "PAUSED":
            status.is_rendering = False
            self.app_state.set_render_status(status)
            self.app_state.status_message.emit(
                f"Render paused at frame {status.last_committed_frame}; Resume available"
            )
            self.render_stopped.emit()
            return
        if status.state == "CANCELLED":
            status.is_rendering = False
            self.app_state.set_render_status(status)
            self.app_state.status_message.emit("Render cancelled")
            self.render_stopped.emit()
            return
        status.is_rendering = False
        if success:
            status.state = "COMPLETE"
            status.progress_percent = 100.0
            self.app_state.status_message.emit("Render completed successfully")
        else:
            if status.state not in {"FAILED", "PAUSE_REQUESTED", "CANCEL_REQUESTED"}:
                status.state = "FAILED"
            if status.state == "PAUSE_REQUESTED":
                status.state = "PAUSED"
            elif status.state == "CANCEL_REQUESTED":
                status.state = "CANCELLED"
            self.app_state.status_message.emit("Render failed" if status.state == "FAILED" else f"Render {status.state.lower()}")
        self.app_state.set_render_status(status)
        self.render_completed.emit(success)

    @Slot(str)
    def _on_error(self, message: str):
        self._progress_timer.stop()
        status = self.app_state.render_status
        status.is_rendering = False
        status.state = "FAILED"
        status.error_message = message
        self.app_state.set_render_status(status)
        self.app_state.status_message.emit(f"Render error: {message}")

    @staticmethod
    def _get_vram_usage() -> float:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage_mb = mem_info.used / (1024 * 1024)
            pynvml.nvmlShutdown()
            return usage_mb
        except Exception:
            return 0.0
