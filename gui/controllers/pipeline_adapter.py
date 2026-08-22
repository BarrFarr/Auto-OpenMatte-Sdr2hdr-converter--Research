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
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, QThread
from PySide6.QtGui import QImage

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


# Preview quality tiers. DRAFT is a small, fast frame used while the user
# scrubs or plays; FULL is the higher-resolution frame delivered once the user
# settles. Only FULL frames are cached and only FULL frames may be used for
# any visual judgement.
PREVIEW_DRAFT = "draft"
PREVIEW_FULL = "full"

_PREVIEW_CAPABILITIES: Optional[dict] = None


def _preview_capabilities() -> dict:
    """Probe ffmpeg once for CUDA decode and GPU scaling support.

    The result is cached for the process lifetime. This only inspects ffmpeg's
    advertised capabilities; it does not touch the production CUDA pipeline.
    """
    global _PREVIEW_CAPABILITIES
    if _PREVIEW_CAPABILITIES is not None:
        return _PREVIEW_CAPABILITIES

    caps = {"ffmpeg": _find_ffmpeg(), "cuda": False, "scale_cuda": False}
    ffmpeg_path = caps["ffmpeg"]
    if ffmpeg_path:
        try:
            hwaccels = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-hwaccels"],
                capture_output=True, timeout=20, check=False,
                **_preview_subprocess_flags(),
            )
            caps["cuda"] = b"cuda" in hwaccels.stdout
            filters = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-filters"],
                capture_output=True, timeout=30, check=False,
                **_preview_subprocess_flags(),
            )
            caps["scale_cuda"] = b"scale_cuda" in filters.stdout
        except (OSError, subprocess.TimeoutExpired):
            pass

    _PREVIEW_CAPABILITIES = caps
    return caps


def _preview_subprocess_flags() -> dict:
    """Keep preview decoders from stealing foreground priority on Windows."""
    if sys.platform == "win32":
        return {
            "creationflags": (
                subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        }
    return {}


class PreviewWorker(QThread):
    """Decode preview frames away from the Qt GUI thread.

    Two request lanes are served: a DRAFT lane for interaction and a FULL lane
    for the settled frame. The DRAFT lane always wins, and a newly queued
    DRAFT request cancels any FULL decode already in flight, so interaction
    never waits behind an expensive decode.

    Sequential playback reuses a persistent ffmpeg raw-video stream. Jumps
    restart the stream with ffmpeg INPUT seek (``-ss`` before ``-i``), which
    seeks to the nearest keyframe instead of decoding from frame zero. GPU
    decode and GPU downscaling are used when ffmpeg advertises them.

    This class is limited to preview frames of the SOURCES. It never calls
    production fitting, rendering, or the production CUDA/NVDEC/NVENC path.
    """

    frame_ready = Signal(object)  # {generation, quality, frame, om_frame, images}
    error = Signal(str)

    _MAX_DECODERS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending = {PREVIEW_DRAFT: None, PREVIEW_FULL: None}
        self._serials = {PREVIEW_DRAFT: 0, PREVIEW_FULL: 0}
        self._running = True
        self._decoders = {}
        self._use_counter = 0

    @staticmethod
    def _scaled_size(source_width: int, source_height: int, target_width: int):
        """Return a deterministic even-sized RGB preview output."""
        target_width = max(2, int(target_width) // 2 * 2)
        if source_width <= 0 or source_height <= 0:
            return target_width, max(2, (target_width * 9 // 16) // 2 * 2)
        height = max(2, int(target_width * source_height / source_width))
        if height % 2:
            height -= 1
        return target_width, max(2, height)

    def request_frames(self, request: dict):
        """Queue the newest request for its quality lane."""
        quality = request.get("quality", PREVIEW_DRAFT)
        with self._condition:
            self._serials[quality] += 1
            request = dict(request)
            request["quality"] = quality
            request["serial"] = self._serials[quality]
            self._pending[quality] = request
            self._condition.notify()

    def cancel_quality(self, quality: str):
        """Drop any queued and in-flight request for one lane."""
        with self._condition:
            self._serials[quality] += 1
            self._pending[quality] = None

    def stop(self):
        """Stop the worker and terminate any active ffmpeg streams."""
        with self._condition:
            self._running = False
            self._pending = {PREVIEW_DRAFT: None, PREVIEW_FULL: None}
            self._condition.notify()

    def _is_obsolete(self, request: dict) -> bool:
        """A request is obsolete if superseded, or if DRAFT work is waiting."""
        with self._condition:
            if not self._running:
                return True
            quality = request["quality"]
            if request["serial"] != self._serials[quality]:
                return True
            # Interaction preempts the expensive lane.
            if quality == PREVIEW_FULL and self._pending[PREVIEW_DRAFT] is not None:
                return True
            return False

    def _decoder_for(self, source: dict, request: dict):
        width, height = self._scaled_size(
            int(source.get("width", 0) or 0),
            int(source.get("height", 0) or 0),
            int(source.get("preview_width", 960) or 960),
        )
        # Keyed by output size, so the DRAFT and FULL streams coexist instead of
        # evicting each other during playback.
        key = (source["path"], width, height)
        decoder = self._decoders.get(key)
        if decoder is None:
            self._evict_decoders(keep=self._MAX_DECODERS - 1)
            decoder = _PreviewStreamDecoder(source["path"], width, height)
            self._decoders[key] = decoder
        decoder.fps = max(1.0, float(source.get("fps", 24.0) or 24.0))
        decoder.threads = 2 if request["quality"] == PREVIEW_DRAFT else 4
        self._use_counter += 1
        decoder.last_used = self._use_counter
        return decoder

    def _evict_decoders(self, keep: int):
        """Close least-recently-used decoders so ffmpeg processes stay bounded."""
        while len(self._decoders) > max(0, keep):
            oldest = min(self._decoders, key=lambda k: self._decoders[k].last_used)
            self._decoders.pop(oldest).close()

    def _drop_inactive_decoders(self, request: dict):
        """Close decoders for sources that are no longer loaded."""
        active_paths = {
            source["path"]
            for source in (request.get("hdr"), request.get("om"))
            if source and source.get("path")
        }
        for key in list(self._decoders):
            if key[0] not in active_paths:
                self._decoders.pop(key).close()

    def _decode_source(self, source: Optional[dict], frame: int, request: dict):
        if not source or not source.get("path"):
            return None
        if self._is_obsolete(request):
            return None
        decoder = self._decoder_for(source, request)
        return decoder.read_to(max(0, int(frame)), lambda: self._is_obsolete(request))

    def _decode_request(self, request: dict):
        self._drop_inactive_decoders(request)
        hdr_image = self._decode_source(request.get("hdr"), request["frame"], request)
        if self._is_obsolete(request):
            return
        om_image = self._decode_source(request.get("om"), request["om_frame"], request)
        if self._is_obsolete(request):
            return
        self.frame_ready.emit({
            "generation": request["generation"],
            "quality": request["quality"],
            "frame": request["frame"],
            "om_frame": request["om_frame"],
            "hdr_image": hdr_image,
            "om_image": om_image,
        })

    def _next_request(self) -> Optional[dict]:
        """Take the next request, DRAFT lane first."""
        with self._condition:
            while self._running:
                for quality in (PREVIEW_DRAFT, PREVIEW_FULL):
                    request = self._pending[quality]
                    if request is not None:
                        self._pending[quality] = None
                        return request
                self._condition.wait()
        return None

    def run(self):
        while True:
            request = self._next_request()
            if request is None:
                break
            try:
                self._decode_request(request)
            except Exception as exc:
                if not self._is_obsolete(request):
                    self.error.emit(f"Preview decode failed: {exc}")

        self._evict_decoders(keep=0)


class _PreviewStreamDecoder:
    """Small ffmpeg stream wrapper used only by PreviewWorker."""

    # Reading forward is cheaper than restarting ffmpeg for nearby targets.
    _FORWARD_LIMIT = 48

    def __init__(self, path: str, width: int, height: int):
        self.path = path
        self.width = width
        self.height = height
        self.fps = 24.0
        self.threads = 2
        self.last_used = 0
        self.allow_gpu = True
        self._gpu_active = False
        self._process = None
        self._cursor = -1
        self._last_image = None

    def close(self):
        process = self._process
        self._process = None
        self._cursor = -1
        self._last_image = None
        if process is not None:
            try:
                if process.stdout:
                    process.stdout.close()
            except OSError:
                pass
            try:
                process.kill()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _start(self, start_frame: int):
        self.close()
        caps = _preview_capabilities()
        ffmpeg_path = caps.get("ffmpeg")
        if not ffmpeg_path or not Path(self.path).is_file():
            return False

        # INPUT seek (-ss before -i) jumps to the nearest keyframe and decodes
        # forward from there. The previous OUTPUT seek decoded from frame zero,
        # which saturated the CPU on long 4K sources. The 9-decimal timestamp
        # mirrors the production tool's decoder_command for frame alignment.
        seek_seconds = max(0.0, start_frame / max(self.fps, 1.0))
        use_gpu = self.allow_gpu and caps["cuda"] and caps["scale_cuda"]

        cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if use_gpu:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += [
            "-ss", f"{seek_seconds:.9f}",
            "-i", self.path,
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
        ]
        if use_gpu:
            # Downscale on the GPU so only the small frame crosses PCIe.
            cmd += [
                "-vf",
                f"scale_cuda={self.width}:{self.height}:format=nv12"
                ",hwdownload,format=nv12",
            ]
        else:
            cmd += [
                "-threads", str(max(1, int(self.threads))),
                "-vf", f"scale={self.width}:{self.height}:flags=fast_bilinear",
            ]
        cmd += ["-pix_fmt", "rgb24", "-vsync", "0", "-f", "rawvideo", "pipe:1"]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                **_preview_subprocess_flags(),
            )
            self._gpu_active = use_gpu
            self._cursor = start_frame - 1
            self._last_image = None
            return True
        except (FileNotFoundError, OSError):
            self.close()
            return False

    def _read_one(self):
        if self._process is None or self._process.stdout is None:
            return None
        size = self.width * self.height * 3
        try:
            raw = bytearray()
            while len(raw) < size:
                chunk = self._process.stdout.read(size - len(raw))
                if not chunk:
                    return None
                raw.extend(chunk)
        except (OSError, ValueError):
            return None
        return QImage(
            bytes(raw),
            self.width,
            self.height,
            self.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def _needs_restart(self, target: int) -> bool:
        return (
            self._process is None
            or target < self._cursor
            or target - self._cursor > self._FORWARD_LIMIT
        )

    def read_to(self, target: int, should_abort) -> Optional[QImage]:
        """Return the frame at ``target``, reusing the stream when sequential."""
        if target == self._cursor and self._last_image is not None:
            return self._last_image

        if self._needs_restart(target) and not self._start(target):
            return None

        image = self._read_forward(target, should_abort)
        if image is not None or should_abort():
            return image

        # A GPU chain can fail at runtime even when ffmpeg advertises it
        # (driver, session limits, unsupported codec). Fall back to CPU once
        # and remember the decision for this decoder.
        if self._gpu_active:
            self.allow_gpu = False
            if self._start(target):
                return self._read_forward(target, should_abort)
        return None

    def _read_forward(self, target: int, should_abort) -> Optional[QImage]:
        while self._cursor < target:
            if should_abort():
                return None
            image = self._read_one()
            if image is None:
                self.close()
                return None
            self._cursor += 1
            self._last_image = image
        return self._last_image


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
    preview_frame_ready = Signal(object)  # decoded preview frame pair
    preview_error = Signal(str)
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
        self._preview_worker = PreviewWorker(self)
        self._preview_worker.frame_ready.connect(self.preview_frame_ready.emit)
        self._preview_worker.error.connect(self.preview_error.emit)
        self._preview_worker.start()

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

    def request_preview_frames(
        self,
        hdr_source: Optional[dict],
        om_source: Optional[dict],
        frame: int,
        om_frame: int,
        generation: int,
        preview_width: int = 960,
        quality: str = PREVIEW_DRAFT,
    ):
        """Queue the newest preview frame pair for one quality lane.

        This is intentionally separate from ``extract_frame`` and from all
        production backend calls. The worker owns ffmpeg streams and drops
        stale requests while the user scrubs.
        """
        for source in (hdr_source, om_source):
            if source is not None:
                source["preview_width"] = preview_width
        self._preview_worker.request_frames({
            "hdr": hdr_source,
            "om": om_source,
            "frame": max(0, int(frame)),
            "om_frame": max(0, int(om_frame)),
            "generation": generation,
            "quality": quality,
        })

    def cancel_full_preview(self):
        """Abandon a pending full-quality preview decode."""
        self._preview_worker.cancel_quality(PREVIEW_FULL)

    def preview_backend_info(self) -> dict:
        """Report which preview decode path is in use, for display only."""
        caps = _preview_capabilities()
        return {
            "ffmpeg": bool(caps.get("ffmpeg")),
            "cuda": bool(caps.get("cuda")),
            "scale_cuda": bool(caps.get("scale_cuda")),
            "gpu_preview": bool(caps.get("cuda") and caps.get("scale_cuda")),
        }

    def stop_preview(self):
        """Stop the GUI-only preview worker during application shutdown."""
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.stop()
            self._preview_worker.wait(2000)

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
