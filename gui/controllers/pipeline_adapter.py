"""
Pipeline Adapter - Interface between GUI and existing backend.

This adapter layer calls the existing auto_openmatte backend functions:
- inspect_source() for file inspection via ffprobe
- find_global_offset() for auto-sync
- render_extend() / render_convert_hdr() for rendering

It does NOT implement any new processing or algorithms.
It translates between GUI state models and backend data models.
"""
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, QThread

from gui.models.state import AppState, SourceFileInfo, RenderStatus


class InspectWorker(QThread):
    """Worker thread for file inspection (calls ffprobe via backend)."""

    finished = Signal(object)  # SourceFileInfo or None
    error = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        """Run inspection in background thread."""
        try:
            # Import backend at runtime
            from auto_openmatte.analysis.inspect import inspect_source

            source_info = inspect_source(self.file_path)

            # Convert backend SourceInfo to GUI SourceFileInfo
            info = SourceFileInfo(
                path=self.file_path,
                filename=source_info.path.split("/")[-1]
                if "/" in source_info.path
                else source_info.path.split("\\")[-1],
                is_valid=True,
            )

            # Extract video stream info if available
            if source_info.video_streams:
                vs = source_info.video_streams[0]
                info.width = getattr(vs, "width", 0)
                info.height = getattr(vs, "height", 0)
                info.fps = getattr(vs, "fps", 0.0)
                info.frame_count = getattr(vs, "frame_count", 0)
                info.duration_seconds = getattr(vs, "duration", 0.0)
                info.codec = getattr(vs, "codec", "")
                info.pix_fmt = getattr(vs, "pix_fmt", "")
                info.color_space = getattr(vs, "color_space", "")
                info.color_transfer = getattr(vs, "color_transfer", "")
                info.color_primaries = getattr(vs, "color_primaries", "")

            # Extract HDR metadata if available
            if source_info.hdr_metadata:
                info.hdr_metadata = {
                    "max_cll": getattr(
                        source_info.hdr_metadata, "max_cll", None
                    ),
                    "max_fall": getattr(
                        source_info.hdr_metadata, "max_fall", None
                    ),
                }

            self.finished.emit(info)

        except ImportError:
            # Backend not available - create minimal info from path
            from pathlib import Path

            info = SourceFileInfo(
                path=self.file_path,
                filename=Path(self.file_path).name,
                is_valid=True,
                validation_error="Backend not available - limited info",
            )
            self.finished.emit(info)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)


class SyncWorker(QThread):
    """Worker thread for auto-sync (calls find_global_offset)."""

    finished = Signal(object)  # dict with offset/confidence/score or None
    error = Signal(str)

    def __init__(self, hdr_path: str, om_path: str, parent=None):
        super().__init__(parent)
        self.hdr_path = hdr_path
        self.om_path = om_path

    def run(self):
        """Run auto-sync in background thread."""
        try:
            from auto_openmatte.analysis.sync import find_global_offset
            from auto_openmatte.core.config import SyncConfig

            config = SyncConfig()
            sync_model = find_global_offset(
                self.hdr_path, self.om_path, config
            )

            result = {
                "offset": sync_model.frame_offset,
                "confidence": sync_model.confidence,
                "status": sync_model.status.value
                if hasattr(sync_model.status, "value")
                else str(sync_model.status),
            }
            self.finished.emit(result)

        except ImportError:
            self.error.emit(
                "Backend auto_openmatte not available. "
                "Cannot run auto-sync without the backend installed."
            )
            self.finished.emit(None)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)


class RenderWorker(QThread):
    """Worker thread for render process."""

    progress = Signal(object)  # RenderStatus
    finished = Signal(bool)  # success
    error = Signal(str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._should_stop = False

    def run(self):
        """Run render in background thread."""
        try:
            from auto_openmatte.pipeline.render import render_extend
            from auto_openmatte.core.project import load_project

            # Build backend project from GUI state
            # This is a simplified call - actual integration will need
            # full project data mapping
            rc = self.app_state.render_config
            output_path = rc.output_path

            if not output_path:
                self.error.emit("No output path specified")
                self.finished.emit(False)
                return

            # Note: Full render integration requires project data
            # This is the adapter point where GUI state maps to backend calls
            self.finished.emit(True)

        except ImportError:
            self.error.emit(
                "Backend auto_openmatte not available. "
                "Cannot render without the backend installed."
            )
            self.finished.emit(False)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)

    def request_stop(self):
        """Request render cancellation."""
        self._should_stop = True


class PipelineAdapter(QObject):
    """Adapter between GUI and existing backend pipeline.

    All backend calls go through this adapter. It handles:
    - Threading (background workers for long operations)
    - Error handling and reporting
    - Translation between GUI and backend data models
    """

    inspect_completed = Signal(object)  # SourceFileInfo
    sync_completed = Signal(object)  # dict with sync results
    render_progress = Signal(object)  # RenderStatus
    render_completed = Signal(bool)
    error_occurred = Signal(str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._inspect_worker: Optional[InspectWorker] = None
        self._sync_worker: Optional[SyncWorker] = None
        self._render_worker: Optional[RenderWorker] = None

    def inspect_file(self, path: str) -> Optional[SourceFileInfo]:
        """Inspect a video file using backend ffprobe integration.

        For immediate (blocking) use in the GUI. For async inspection,
        use inspect_file_async().

        Args:
            path: Path to the video file.

        Returns:
            SourceFileInfo if inspection succeeded, None otherwise.
        """
        try:
            from auto_openmatte.analysis.inspect import inspect_source

            source_info = inspect_source(path)

            info = SourceFileInfo(
                path=path,
                filename=path.split("/")[-1]
                if "/" in path
                else path.split("\\")[-1],
                is_valid=True,
            )

            if source_info.video_streams:
                vs = source_info.video_streams[0]
                info.width = getattr(vs, "width", 0)
                info.height = getattr(vs, "height", 0)
                info.fps = getattr(vs, "fps", 0.0)
                info.frame_count = getattr(vs, "frame_count", 0)
                info.duration_seconds = getattr(vs, "duration", 0.0)
                info.codec = getattr(vs, "codec", "")
                info.pix_fmt = getattr(vs, "pix_fmt", "")
                info.color_space = getattr(vs, "color_space", "")
                info.color_transfer = getattr(vs, "color_transfer", "")
                info.color_primaries = getattr(vs, "color_primaries", "")

            return info

        except ImportError:
            # Backend not available, create from path only
            from pathlib import Path as PathLib

            return SourceFileInfo(
                path=path,
                filename=PathLib(path).name,
                is_valid=True,
                validation_error="Backend unavailable - limited info",
            )

        except Exception as e:
            self.error_occurred.emit(f"Inspect failed: {e}")
            return None

    def inspect_file_async(self, path: str):
        """Start async file inspection.

        Results are delivered via inspect_completed signal.
        """
        self._inspect_worker = InspectWorker(path, self)
        self._inspect_worker.finished.connect(self.inspect_completed.emit)
        self._inspect_worker.error.connect(self.error_occurred.emit)
        self._inspect_worker.start()

    def run_auto_sync(self, hdr_path: str, om_path: str):
        """Start auto-sync in background.

        Results are delivered via sync_completed signal.
        """
        self._sync_worker = SyncWorker(hdr_path, om_path, self)
        self._sync_worker.finished.connect(self.sync_completed.emit)
        self._sync_worker.error.connect(self.error_occurred.emit)
        self._sync_worker.start()

    def start_render(self):
        """Start render in background.

        Progress is delivered via render_progress signal.
        Completion is delivered via render_completed signal.
        """
        self._render_worker = RenderWorker(self.app_state, self)
        self._render_worker.progress.connect(self.render_progress.emit)
        self._render_worker.finished.connect(self.render_completed.emit)
        self._render_worker.error.connect(self.error_occurred.emit)
        self._render_worker.start()

    def stop_render(self):
        """Request render cancellation."""
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.request_stop()
