"""
Render Controller - Manages the render process lifecycle.

Handles:
- Starting/stopping render via backend adapter
- Progress polling and VRAM monitoring
- ETA calculation
- Error reporting

Does NOT implement rendering logic. Calls backend render functions.
"""
import time
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from gui.models.state import AppState, RenderStatus
from gui.controllers.pipeline_adapter import PipelineAdapter


class RenderController(QObject):
    """Controls render lifecycle and progress monitoring."""

    render_started = Signal()
    render_stopped = Signal()
    render_completed = Signal(bool)  # success

    def __init__(
        self, app_state: AppState, adapter: PipelineAdapter, parent=None
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.adapter = adapter

        self._start_time: Optional[float] = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(500)  # 500ms polling
        self._progress_timer.timeout.connect(self._poll_progress)

        # Connect adapter signals
        self.adapter.render_completed.connect(self._on_render_finished)
        self.adapter.render_progress.connect(self._on_progress_update)
        self.adapter.error_occurred.connect(self._on_error)

    def start_render(self):
        """Start the render process."""
        rc = self.app_state.render_config

        if not rc.output_path:
            self.app_state.status_message.emit(
                "Cannot render: no output path specified"
            )
            return

        # Check prerequisites
        if not self.app_state.hdr_source or not self.app_state.om_source:
            self.app_state.status_message.emit(
                "Cannot render: both sources required"
            )
            return

        # Check that at least one shot is locked
        locked = [sl for sl in self.app_state.shot_locks if sl.is_locked]
        if not locked and self.app_state.shot_locks:
            self.app_state.status_message.emit(
                "Cannot render: no shots are locked"
            )
            return

        # Initialize render status
        total_frames = 0
        if self.app_state.hdr_source:
            total_frames = self.app_state.hdr_source.frame_count

        status = RenderStatus(
            is_rendering=True,
            progress_percent=0.0,
            current_frame=0,
            total_frames=total_frames,
            fps=0.0,
            eta_seconds=0.0,
            elapsed_seconds=0.0,
            vram_usage_mb=0.0,
            output_path=rc.output_path,
            error_message="",
        )
        self.app_state.set_render_status(status)

        self._start_time = time.time()
        self._progress_timer.start()
        self.render_started.emit()

        self.app_state.status_message.emit(
            f"Rendering to: {rc.output_path}"
        )

        # Start backend render
        self.adapter.start_render()

    def stop_render(self):
        """Stop the current render."""
        self._progress_timer.stop()
        self.adapter.stop_render()

        status = self.app_state.render_status
        status.is_rendering = False
        self.app_state.set_render_status(status)

        self.render_stopped.emit()
        self.app_state.status_message.emit("Render stopped by user")

    @Slot()
    def _poll_progress(self):
        """Poll render progress and update status."""
        if not self._start_time:
            return

        elapsed = time.time() - self._start_time
        status = self.app_state.render_status
        status.elapsed_seconds = elapsed

        # Calculate ETA from current progress
        if status.progress_percent > 0:
            total_estimated = elapsed / (status.progress_percent / 100.0)
            status.eta_seconds = max(0.0, total_estimated - elapsed)

        # Calculate FPS
        if elapsed > 0 and status.current_frame > 0:
            status.fps = status.current_frame / elapsed

        # Try to get VRAM usage
        status.vram_usage_mb = self._get_vram_usage()

        self.app_state.set_render_status(status)

    @Slot(object)
    def _on_progress_update(self, progress_data):
        """Handle progress update from render worker."""
        if isinstance(progress_data, RenderStatus):
            self.app_state.set_render_status(progress_data)

    @Slot(bool)
    def _on_render_finished(self, success: bool):
        """Handle render completion."""
        self._progress_timer.stop()

        status = self.app_state.render_status
        status.is_rendering = False
        if success:
            status.progress_percent = 100.0
        self.app_state.set_render_status(status)

        self.render_completed.emit(success)

        if success:
            self.app_state.status_message.emit("Render completed successfully")
        else:
            self.app_state.status_message.emit("Render failed")

    @Slot(str)
    def _on_error(self, message: str):
        """Handle render errors."""
        self._progress_timer.stop()

        status = self.app_state.render_status
        status.is_rendering = False
        status.error_message = message
        self.app_state.set_render_status(status)

        self.app_state.status_message.emit(f"Render error: {message}")

    @staticmethod
    def _get_vram_usage() -> float:
        """Attempt to get VRAM usage in MB.

        Uses pynvml if available, otherwise returns 0.
        This is a monitoring-only call, does not affect rendering.
        """
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
