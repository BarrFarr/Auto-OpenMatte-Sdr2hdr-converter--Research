"""
Pipeline Adapter - Interface between GUI and existing backend.

This adapter layer calls the existing auto_openmatte backend functions:
- inspect_source() for file inspection via ffprobe
- find_global_offset() for auto-sync
- render_extend() / render_convert_hdr() for rendering
- extract_frame() for single-frame preview via ffmpeg subprocess

It does NOT implement any new processing or algorithms.
It translates between GUI state models and backend data models.
"""
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage

from gui.models.state import (
    AppState,
    AudioTrackInfo,
    RenderStatus,
    SourceFileInfo,
)

logger = logging.getLogger(__name__)


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


def _probe_audio_tracks(path: str) -> list[AudioTrackInfo]:
    """Probe selectable audio streams without changing video inspection."""
    try:
        from auto_openmatte.output.mux import probe_audio_tracks

        return [AudioTrackInfo(
            ordinal=int(track.get("ordinal", 0)),
            global_index=int(track.get("global_index", 0)),
            codec_name=str(track.get("codec_name", "")),
            codec_long_name=str(track.get("codec_long_name", "")),
            channels=int(track.get("channels", 0)),
            channel_layout=str(track.get("channel_layout", "")),
            sample_rate=int(track.get("sample_rate", 0)),
            language=str(track.get("language", "")),
            title=str(track.get("title", "")),
            is_default=bool(track.get("default", False)),
        ) for track in probe_audio_tracks(Path(path))]
    except Exception:
        return []


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
    info.audio_tracks = _probe_audio_tracks(path)
    return info


def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg executable.

    Search order:
    1. Backend MediaToolLocator (bundled / dev tree / PATH)
    2. shutil.which on PATH
    3. Hardcoded Windows locations
    """
    # Try backend locator first
    try:
        from auto_openmatte.utils.ffmpeg import get_media_tool_config

        config = get_media_tool_config()
        if config.ffmpeg_found and config.ffmpeg_path is not None:
            return str(config.ffmpeg_path)
    except (ImportError, Exception):
        pass

    # Fallback: shutil.which
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Fallback: common Windows paths
    if sys.platform == "win32":
        candidates = [
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"),
            Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe",
        ]
        # Also check relative dev path from this file
        repo_root_candidate = Path(__file__).resolve().parent.parent.parent
        dev_ffmpeg = repo_root_candidate / "dev" / "ffmpeg-build" / "install" / "bin" / "ffmpeg.exe"
        candidates.insert(0, dev_ffmpeg)

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

    return None


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
                "frame_locked": bool(getattr(sync_model, "frame_locked", False)),
                "status": sync_model.status.value
                if hasattr(sync_model.status, "value")
                else str(sync_model.status),
            }
            self.finished.emit(result)

        except ImportError as exc:
            logger.exception("Auto-sync backend import failed")
            self.error.emit(f"Auto-sync backend import failed: {exc}")
            self.finished.emit(None)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)


class FastSyncWorker(QThread):
    """Worker for the optional first-ten-minute Fast Auto Sync strategy."""

    completed = Signal(object)
    progress = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        hdr_path: str,
        om_path: str,
        analysis_seconds: float = 600.0,
        parent=None,
    ):
        super().__init__(parent)
        self.hdr_path = hdr_path
        self.om_path = om_path
        self.analysis_seconds = float(analysis_seconds)

    def run(self):
        """Run bounded fast sync in the background thread."""
        started = time.perf_counter()

        def report(message: str):
            self.progress.emit(
                {
                    "strategy": "fast",
                    "message": message,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

        try:
            from auto_openmatte.analysis.gpu_fast_sync import (
                find_fast_global_offset_gpu,
            )
            from auto_openmatte.core.config import SyncConfig

            hdr_source = _inspect_backend_source(self.hdr_path)
            om_source = _inspect_backend_source(self.om_path)
            config = SyncConfig(analysis_range_seconds=self.analysis_seconds)
            sync_model = find_fast_global_offset_gpu(
                hdr_source,
                om_source,
                config,
                progress_callback=report,
            )
            elapsed_seconds = time.perf_counter() - started
            result = {
                "offset": sync_model.frame_offset,
                "confidence": sync_model.confidence,
                "frame_locked": bool(getattr(sync_model, "frame_locked", False)),
                "status": sync_model.status.value
                if hasattr(sync_model.status, "value")
                else str(sync_model.status),
                "strategy": "fast",
                "backend": "cuda-v05-native",
                "gpu_only": True,
                "method": getattr(sync_model, "method", ""),
                "score": float(
                    (getattr(sync_model, "fast_diagnostics", {}) or {}).get(
                        "best_score", sync_model.confidence
                    )
                ),
                "elapsed_seconds": elapsed_seconds,
                "analysis_range_seconds": self.analysis_seconds,
                "fast_confidence": getattr(sync_model, "fast_confidence", None),
                "fast_diagnostic_status": getattr(
                    sync_model, "fast_diagnostic_status", ""
                ),
                "fast_diagnostics": getattr(sync_model, "fast_diagnostics", {}),
            }
            self.completed.emit(result)

        except ImportError as exc:
            logger.exception("Fast auto-sync backend import failed")
            self.error.emit(f"Fast auto-sync backend import failed: {exc}")

        except Exception as exc:
            self.error.emit(str(exc))


class RenderWorker(QThread):
    """Worker for the checkpointed segmented canonical renderer."""

    progress = Signal(object)  # RenderStatus
    finished = Signal(bool)  # success
    error = Signal(str)

    def __init__(self, app_state: AppState, *, resume: bool = False, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._resume = bool(resume)
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()

    def _emit_progress(
        self,
        current_frame: int,
        total_frames: int,
        mode: str,
        output_path: str,
        meta: dict | None = None,
    ):
        meta = meta or {}
        percent = (100.0 * current_frame / total_frames) if total_frames else 0.0
        self.progress.emit(
            RenderStatus(
                is_rendering=meta.get("state") not in {"PAUSED", "COMPLETE", "CANCELLED", "FAILED"},
                progress_percent=percent,
                current_frame=current_frame,
                total_frames=total_frames,
                output_path=output_path,
                processing_mode=mode,
                state=str(meta.get("state", "RUNNING")),
                current_segment=meta.get("current_segment"),
                total_segments=int(meta.get("total_segments", 0) or 0),
                last_committed_frame=int(meta.get("last_committed_frame", -1) or -1),
                checkpoint_path=str(meta.get("checkpoint_path", "")),
            )
        )

    def run(self):
        """Run one new or resumable checkpointed render job."""
        success = False
        try:
            from auto_openmatte.core.config import RenderConfig as BackendRenderConfig
            from auto_openmatte.core.mode import ProcessingMode
            from auto_openmatte.core.project import load_project
            from auto_openmatte.pipeline.render_job import run_segmented_render

            rc = self.app_state.render_config
            output_path = rc.output_path
            if not output_path:
                raise ValueError("No output path specified")

            backend_project = Path(rc.backend_project_path) if rc.backend_project_path else None
            if backend_project is None and self.app_state.project_path:
                sibling = Path(self.app_state.project_path).with_suffix(".json")
                if sibling.is_file():
                    backend_project = sibling
            if backend_project is None or not backend_project.is_file():
                raise ValueError(
                    "No analyzed backend project.json selected. Set Backend Project "
                    "to the canonical analysis project before rendering."
                )

            mode = ProcessingMode.coerce(self.app_state.processing_mode)
            project = load_project(backend_project)
            project.processing_mode = mode
            backend_config = BackendRenderConfig(
                codec=rc.codec,
                crf=rc.crf,
                pix_fmt=rc.pix_fmt,
                resolution=rc.resolution,
                processing_mode=mode,
                transform=project.transform_config,
            )
            # These lifecycle options are orchestration-only and are not read
            # by fitting/compositing/CUDA code.
            backend_config.segment_frames = max(1, int(rc.segment_frames))
            backend_config.checkpoint_path = rc.checkpoint_path

            audio_type = str(rc.audio_source or "NONE").upper()
            audio_selection = {
                "type": audio_type,
                "ordinal": int(rc.audio_stream_ordinal),
                "global_index": int(rc.audio_stream_index),
                "codec_name": rc.audio_codec,
                "language": rc.audio_language,
                "title": rc.audio_title,
            }
            if audio_type not in {"NONE", "HDR", "OM"}:
                raise ValueError(f"Unsupported audio source: {audio_type}")
            if audio_type != "NONE" and int(rc.audio_stream_ordinal) < 0:
                raise ValueError("Select an audio track before rendering")

            self._emit_progress(0, 0, mode.value, output_path)
            result = run_segmented_render(
                project,
                Path(output_path),
                backend_config,
                project_path=self.app_state.project_path,
                audio_selection=audio_selection,
                resume=self._resume,
                cancel_event=self._cancel_event,
                pause_event=self._pause_event,
                progress_callback=lambda current, total, meta: self._emit_progress(
                    current, total, mode.value, output_path, meta
                ),
            )
            success = bool(result)

        except ImportError as exc:
            self.error.emit(f"Render backend import failed: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit(success)

    def request_stop(self):
        """Request cancellation at the next safe frame/segment boundary."""
        self._cancel_event.set()

    def request_pause(self):
        """Request a pause; the active segment is discarded safely."""
        self._pause_event.set()


class PipelineAdapter(QObject):
    """Adapter between GUI and existing backend pipeline.

    All backend calls go through this adapter. It handles:
    - Threading (background workers for long operations)
    - Error handling and reporting
    - Translation between GUI and backend data models
    """

    inspect_completed = Signal(object)  # SourceFileInfo
    sync_completed = Signal(object)  # dict with sync results
    sync_progress = Signal(object)  # fast-sync progress payload
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
            from auto_openmatte.preview.cuda_backend import CudaPreviewBackend
            from tools.openmatte_hdr.cuda_backend import CUDA_CAPABILITIES

            capabilities = CudaPreviewBackend(device_id=0).capabilities
            declared = CUDA_CAPABILITIES
            return {
                "available": bool(capabilities.available),
                "backend_name": declared.backend,
                "hardware_decode": bool(capabilities.hardware_decode),
                "hardware_encode": declared.hardware_encode,
                "zero_copy": declared.zero_copy,
                "supported_codecs": list(declared.supported_codecs),
                "reason": capabilities.reason,
            }
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: try importing just the contracts module
        try:
            import auto_openmatte.backends  # noqa: F401

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

    def extract_frame(
        self, path: str, frame_number: int, width: int = 960
    ) -> Optional[QImage]:
        """Extract a single video frame as a QImage using ffmpeg subprocess.

        Uses ffmpeg to decode one frame at the given frame number, output as
        RGB24 rawvideo via pipe, and wraps the result in a QImage.

        Args:
            path: Path to the video file.
            frame_number: 0-indexed frame number to extract.
            width: Target width for scaling (height auto-calculated).

        Returns:
            QImage in RGB888 format, or None if extraction fails.
        """
        ffmpeg_path = _find_ffmpeg()
        if ffmpeg_path is None:
            return None

        if not path or not Path(path).is_file():
            return None

        # Build ffmpeg command:
        # -vf select=eq(n,FRAME_NUMBER),scale=WIDTH:-1
        # Output single frame as RGB24 rawvideo to pipe
        vf_filter = f"select=eq(n\\,{frame_number}),scale={width}:-1"

        cmd = [
            ffmpeg_path,
            "-v", "quiet",
            "-nostdin",
            "-i", path,
            "-vf", vf_filter,
            "-frames:v", "1",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0 or not result.stdout:
                return None

            raw_data = result.stdout
            # Calculate height from raw data size: size = width * height * 3
            expected_row_bytes = width * 3
            if len(raw_data) < expected_row_bytes:
                return None

            height = len(raw_data) // expected_row_bytes
            if height <= 0:
                return None

            # Trim any extra bytes (shouldn't happen but be safe)
            expected_size = width * height * 3
            raw_data = raw_data[:expected_size]

            # Create QImage from raw RGB24 data
            image = QImage(
                raw_data,
                width,
                height,
                width * 3,  # bytes per line
                QImage.Format.Format_RGB888,
            )
            # Make a deep copy since raw_data buffer will be freed
            return image.copy()

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
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

    def run_fast_auto_sync(
        self,
        hdr_path: str,
        om_path: str,
        analysis_seconds: float = 600.0,
    ):
        """Start the bounded GPU fast auto-sync strategy."""
        self._sync_worker = FastSyncWorker(
            hdr_path,
            om_path,
            analysis_seconds=analysis_seconds,
            parent=self,
        )
        self._sync_worker.completed.connect(self.sync_completed.emit)
        self._sync_worker.progress.connect(self.sync_progress.emit)
        self._sync_worker.error.connect(self.error_occurred.emit)
        self._sync_worker.start()

    def start_render(self, *, resume: bool = False):
        """Start a new or resumable segmented render job."""
        if self._render_worker is not None and self._render_worker.isRunning():
            self.error_occurred.emit(
                "A render job is already running; wait for it to stop before starting another."
            )
            return
        backend_status = self.get_backend_status()
        if backend_status.get("available"):
            self.app_state.status_message.emit(
                f"Using backend: {backend_status['backend_name']}"
            )

        self._render_worker = RenderWorker(self.app_state, resume=resume, parent=self)
        self._render_worker.progress.connect(self.render_progress.emit)
        self._render_worker.finished.connect(self.render_completed.emit)
        self._render_worker.error.connect(self.error_occurred.emit)
        self._render_worker.start()

    def stop_render(self):
        """Request cancellation at a safe frame boundary."""
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.request_stop()

    def pause_render(self):
        """Request a safe checkpoint pause at the current segment boundary."""
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.request_pause()

    def resume_render(self):
        """Start a fresh worker that reconciles and resumes the checkpoint."""
        self.start_render(resume=True)
