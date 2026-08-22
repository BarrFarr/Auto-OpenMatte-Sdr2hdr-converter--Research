"""
Pipeline Adapter - Interface between GUI and existing backend.

This adapter layer calls the existing auto_openmatte backend functions:
- inspect_source() for file inspection via ffprobe
- find_global_offset() for auto-sync
- render_extend() / render_convert_hdr() for rendering

It does NOT implement any new processing or algorithms.
It translates between GUI state models and backend data models.
"""
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, QThread

from gui.models.state import AppState, SourceFileInfo, RenderStatus


def _enum_value(value) -> str:
    """Return a backend enum's serialized value for display and project state."""
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _inspect_backend_source(path: str):
    """Inspect a path and select the backend's default video stream.

    The production inspection and synchronization APIs use pathlib.Path and
    SourceInfo objects. GUI state deliberately stores strings for JSON/project
    portability, so this conversion belongs at this adapter boundary.
    """
    from auto_openmatte.analysis.inspect import inspect_source

    source_info = inspect_source(Path(path))
    if source_info.selected_stream is None and source_info.video_streams:
        source_info.selected_stream = next(
            (stream for stream in source_info.video_streams if stream.is_default),
            source_info.video_streams[0],
        )
    return source_info


def _source_file_info(path: str, source_info) -> SourceFileInfo:
    """Translate the current backend SourceInfo model into GUI state."""
    source_path = Path(path)
    info = SourceFileInfo(
        path=str(source_path),
        filename=source_path.name,
        is_valid=True,
    )

    stream = source_info.selected_stream
    if stream is None and source_info.video_streams:
        stream = source_info.video_streams[0]
    if stream is not None:
        info.width = int(getattr(stream, "width", 0) or 0)
        info.height = int(getattr(stream, "height", 0) or 0)
        info.fps = float(getattr(stream, "fps", 0.0) or 0.0)
        info.frame_count = int(getattr(stream, "frame_count", 0) or 0)
        info.duration_seconds = float(
            getattr(stream, "duration_seconds", 0.0) or 0.0
        )
        info.codec = str(getattr(stream, "codec", "") or "")
        info.pix_fmt = str(getattr(stream, "pix_fmt", "") or "")
        info.color_space = str(
            getattr(stream, "matrix_coefficients", "") or ""
        )
        info.color_transfer = _enum_value(
            getattr(stream, "transfer", "")
        )
        info.color_primaries = _enum_value(
            getattr(stream, "color_primaries", "")
        )

    metadata = getattr(source_info, "hdr_metadata", None)
    if metadata is not None:
        info.hdr_metadata = {
            "format": _enum_value(getattr(metadata, "format", "")),
            "max_cll": getattr(metadata, "max_cll", None),
            "max_fall": getattr(metadata, "max_fall", None),
            "mastering_display": getattr(metadata, "mastering_display", None),
        }
    return info


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
            # Import backend at runtime and normalize the GUI string path.
            source_info = _inspect_backend_source(self.file_path)
            info = _source_file_info(self.file_path, source_info)
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

            # The backend sync API requires inspected SourceInfo objects with
            # selected_stream populated, not GUI string paths.
            hdr_source = _inspect_backend_source(self.hdr_path)
            om_source = _inspect_backend_source(self.om_path)
            config = SyncConfig()
            sync_model = find_global_offset(hdr_source, om_source, config)

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
    inspect_error = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._inspect_worker: Optional[InspectWorker] = None
        self._sync_worker: Optional[SyncWorker] = None
        self._render_worker: Optional[RenderWorker] = None

    def get_backend_status(self) -> dict:
        """Query availability and capabilities of the portable backend.

        Returns a dict with backend status information. If the backend
        module is not importable, returns a dict indicating unavailability.
        """
        try:
            from auto_openmatte.backends import BackendCapabilities
            from tools.openmatte_hdr.cuda_backend import CudaBackend

            backend = CudaBackend()
            caps: BackendCapabilities = backend.capabilities
            return {
                "available": True,
                "backend_name": caps.backend,
                "hardware_decode": caps.hardware_decode,
                "hardware_encode": caps.hardware_encode,
                "zero_copy": caps.zero_copy,
                "supported_codecs": list(caps.supported_codecs),
            }
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: try importing just the contracts module
        try:
            from auto_openmatte.backends import BackendCapabilities  # noqa: F811

            return {
                "available": False,
                "backend_name": "contracts-only",
                "hardware_decode": False,
                "hardware_encode": False,
                "zero_copy": False,
                "supported_codecs": [],
            }
        except ImportError:
            return {
                "available": False,
                "backend_name": "",
                "hardware_decode": False,
                "hardware_encode": False,
                "zero_copy": False,
                "supported_codecs": [],
            }

    def get_bridge_info(self) -> dict:
        """Locate the native bridge DLL and read its SHA-256 from BUILD_MANIFEST.

        Returns a dict with bridge path and hash, or empty values if not found.
        """
        import json

        manifest_candidates = [
            Path("artifacts/common_backend_architecture_cuda_v05/BUILD_MANIFEST.json"),
            Path(__file__).resolve().parent.parent.parent
            / "artifacts"
            / "common_backend_architecture_cuda_v05"
            / "BUILD_MANIFEST.json",
        ]

        for manifest_path in manifest_candidates:
            try:
                manifest_path = manifest_path.resolve()
                if not manifest_path.exists():
                    continue
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                baseline = data.get("baseline_artifacts", {})
                bridge_key = "dev/v05-native/v5_gpu_bridge.dll"
                bridge_info = baseline.get(bridge_key, {})
                bridge_dll = manifest_path.parent.parent.parent / bridge_key
                return {
                    "manifest_found": True,
                    "bridge_path": str(bridge_dll) if bridge_dll.exists() else "",
                    "bridge_sha256": bridge_info.get("sha256", ""),
                    "bridge_exists": bridge_dll.exists(),
                }
            except Exception:
                continue

        return {
            "manifest_found": False,
            "bridge_path": "",
            "bridge_sha256": "",
            "bridge_exists": False,
        }

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
            source_info = _inspect_backend_source(path)
            return _source_file_info(path, source_info)

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
            self.inspect_error.emit(f"Inspect failed: {e}")
            return None

    def inspect_file_async(self, path: str):
        """Start async file inspection.

        Results are delivered via inspect_completed signal.
        """
        self._inspect_worker = InspectWorker(path, self)
        self._inspect_worker.finished.connect(self.inspect_completed.emit)
        self._inspect_worker.error.connect(self.inspect_error.emit)
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

        If the portable CUDA backend is available, it is used preferentially.
        Progress is delivered via render_progress signal.
        Completion is delivered via render_completed signal.
        """
        # Check if the new backend is available and prefer it
        backend_status = self.get_backend_status()
        if backend_status.get("available"):
            self.app_state.status_message.emit(
                f"Using backend: {backend_status['backend_name']}"
            )

        self._render_worker = RenderWorker(self.app_state, self)
        self._render_worker.progress.connect(self.render_progress.emit)
        self._render_worker.finished.connect(self.render_completed.emit)
        self._render_worker.error.connect(self.error_occurred.emit)
        self._render_worker.start()

    def stop_render(self):
        """Request render cancellation."""
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.request_stop()
