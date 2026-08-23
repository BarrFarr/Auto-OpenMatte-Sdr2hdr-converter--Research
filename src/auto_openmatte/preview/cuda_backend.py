"""CUDA-only preview adapter over existing GPU/native implementations.

This module does not reimplement color science, resize kernels, NVDEC, or
NVENC.  It is intentionally fail-closed when the checkout lacks the native V5
runtime.  The public surface contains only neutral preview contracts; CuPy,
ctypes, and native decoder objects remain private to this adapter.
"""

from __future__ import annotations

import importlib
import os
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from auto_openmatte.backends import (
    Frame,
    FrameMemory,
    FrameTimestamps,
    MemoryDomain,
    Ownership,
    ResampleRequest,
)
from auto_openmatte.core.config import TransformConfig
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.preview.contracts import (
    PreviewBackend,
    PreviewBackendUnavailable,
    PreviewCancellation,
    PreviewCancelled,
    PreviewDecoderSession,
    SingleSourceDecoderBackend,
)
from auto_openmatte.preview.models import (
    PreviewCapabilities,
    PreviewFrame,
    PreviewMode,
    PreviewRequest,
    ResizeRequest,
)


class CudaPreviewBackend(PreviewBackend):
    """CUDA preview boundary with explicit capability/dependency gating."""

    backend_id = "cuda-v05-preview"

    def __init__(
        self,
        *,
        device_id: int = 0,
        bridge_path: Path | None = None,
        decoder_factory: Callable[..., PreviewDecoderSession] | None = None,
    ) -> None:
        self.device_id = int(device_id)
        self.bridge_path = bridge_path or self._default_bridge_path()
        self._decoder_factory = decoder_factory
        self._transform_backend: Any = None
        self._resample_backend: Any = None
        self._cupy: Any = None
        self._ffmpeg_bin: Path | None = None
        self._capabilities = self._probe()

    @staticmethod
    def _workspace_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _runtime_tools_dir(cls) -> Path:
        configured = os.environ.get("OPENMATTE_V5_TOOLS")
        if configured:
            return Path(configured).expanduser().resolve()
        runtime_root = os.environ.get("OPENMATTE_V5_RUNTIME")
        if runtime_root:
            runtime_tools = Path(runtime_root).expanduser() / "tools" / "openmatte_hdr"
            if runtime_tools.is_dir():
                return runtime_tools.resolve()
        return cls._workspace_root() / "tools" / "openmatte_hdr"

    @classmethod
    def _default_bridge_path(cls) -> Path:
        configured = os.environ.get("OPENMATTE_V5_BRIDGE_PATH")
        if configured:
            return Path(configured).expanduser().resolve()
        runtime_root = os.environ.get("OPENMATTE_V5_RUNTIME")
        if runtime_root:
            runtime_bridge = (
                Path(runtime_root).expanduser()
                / "dev"
                / "v05-native"
                / "v5_gpu_bridge.dll"
            )
            if runtime_bridge.is_file():
                return runtime_bridge.resolve()
        return cls._workspace_root() / "dev" / "v05-native" / "v5_gpu_bridge.dll"

    @classmethod
    def _import_native_module(cls) -> Any:
        """Import the existing V5 module with its sibling-module layout."""
        try:
            return importlib.import_module("v05_gpu_native")
        except ModuleNotFoundError:
            tools_dir = cls._runtime_tools_dir()
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            return importlib.import_module("v05_gpu_native")

    @staticmethod
    def _fps_fraction(value: tuple[int, int] | str | float) -> tuple[int, int]:
        if isinstance(value, tuple):
            fraction = Fraction(int(value[0]), int(value[1]))
        else:
            fraction = Fraction(str(value))
        if fraction <= 0:
            raise ValueError("source fps must be positive")
        return int(fraction.numerator), int(fraction.denominator)

    @classmethod
    def _source_fps(cls, source_path: Path) -> tuple[int, int]:
        """Read the source rate through the existing FFmpeg resolver."""
        from auto_openmatte.utils.ffmpeg import get_media_tool_config

        config = get_media_tool_config()
        ffprobe = config.require("ffprobe")
        import json
        import subprocess

        result = subprocess.run(
            [
                str(ffprobe),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                "-of", "json",
                str(source_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise PreviewBackendUnavailable(
                f"ffprobe failed for {source_path}: {result.stderr.strip()}"
            )
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [None])[0]
        if not stream:
            raise PreviewBackendUnavailable(f"no video stream found in {source_path}")
        for key in ("r_frame_rate", "avg_frame_rate"):
            value = str(stream.get(key) or "0/0")
            if value not in {"0/0", "0", ""}:
                return cls._fps_fraction(value)
        raise PreviewBackendUnavailable(f"video stream has no usable fps: {source_path}")

    def _probe(self) -> PreviewCapabilities:
        missing: list[str] = []
        try:
            cupy = importlib.import_module("cupy")
            if int(cupy.cuda.runtime.getDeviceCount()) <= self.device_id:
                missing.append(f"CUDA device {self.device_id} unavailable")
            else:
                self._cupy = cupy
        except Exception as exc:
            missing.append(f"CuPy/CUDA unavailable: {exc}")

        try:
            importlib.import_module("numpy")
        except Exception as exc:
            missing.append(f"NumPy unavailable: {exc}")

        tools_dir = self._runtime_tools_dir()
        if not (tools_dir / "v05_streaming.py").is_file():
            missing.append(f"V5 runtime streaming module is missing: {tools_dir / 'v05_streaming.py'}")
        if not self.bridge_path.is_file():
            missing.append(f"native bridge DLL is missing: {self.bridge_path}")

        try:
            from auto_openmatte.utils.ffmpeg import get_media_tool_config

            media_tools = get_media_tool_config()
            if not media_tools.complete:
                missing.append("FFmpeg and FFprobe are not both available")
            elif media_tools.ffmpeg_path is not None:
                self._ffmpeg_bin = media_tools.ffmpeg_path.parent
        except Exception as exc:
            missing.append(f"FFmpeg resolver unavailable: {exc}")

        # cuda_backend.py is available as an adapter, but native V5 imports
        # remain gated because v05_gpu_native imports the deployment module.
        try:
            importlib.import_module("tools.openmatte_hdr.cuda_backend")
        except Exception as exc:
            missing.append(f"CUDA facade unavailable: {exc}")

        if not missing and (tools_dir / "v05_streaming.py").is_file():
            try:
                self._import_native_module()
            except Exception as exc:
                missing.append(f"V5 native module unavailable: {exc}")

        available = not missing
        return PreviewCapabilities(
            backend_id=self.backend_id,
            available=available,
            hardware_decode=available,
            gpu_resize=available,
            gpu_transform=available,
            gpu_preview=available,
            reason="; ".join(missing),
        )

    @property
    def capabilities(self) -> PreviewCapabilities:
        return self._capabilities

    def _require_available(self) -> None:
        if not self.capabilities.available:
            raise PreviewBackendUnavailable(
                f"{self.backend_id} is unavailable in this checkout: "
                f"{self.capabilities.reason}"
            )

    def _default_decoder_session(
        self,
        source_path: Path,
        *,
        source_role: PreviewMode,
        locked_offset: int,
    ) -> PreviewDecoderSession:
        if self._ffmpeg_bin is None:
            raise PreviewBackendUnavailable("FFmpeg runtime directory is unavailable")
        fps = self._source_fps(Path(source_path))
        is_hdr = source_role == PreviewMode.SOURCE_HDR
        decoder = "hevc_cuvid" if is_hdr else "h264_cuvid"
        start_frame = 0 if is_hdr else int(locked_offset)
        native_module = self._import_native_module()
        source_mode = native_module.HDR_MODE if is_hdr else native_module.SDR_MODE
        native = CudaSingleSourceDecoderBackend(
            source_mode=source_mode,
            ffmpeg_bin=self._ffmpeg_bin,
            bridge_path=self.bridge_path,
            device_index=self.device_id,
        )
        return CudaSingleSourceDecoderSession(
            native,
            source_path=Path(source_path),
            decoder=decoder,
            start_frame=start_frame,
            fps=fps,
            source_role=source_role,
        )

    def create_decoder_session(
        self,
        source_path: Path,
        *,
        source_role: PreviewMode,
        locked_offset: int,
    ) -> PreviewDecoderSession:
        self._require_available()
        if self._decoder_factory is not None:
            return self._decoder_factory(
                Path(source_path),
                source_role=source_role,
                locked_offset=int(locked_offset),
                device_id=self.device_id,
                bridge_path=self.bridge_path,
            )
        return self._default_decoder_session(
            Path(source_path),
            source_role=source_role,
            locked_offset=int(locked_offset),
        )

    def _require_transform_backend(self) -> Any:
        self._require_available()
        if self._transform_backend is None:
            from auto_openmatte.processing.transform_backend import GPUTransformBackend

            self._transform_backend = GPUTransformBackend(device_id=self.device_id)
        return self._transform_backend

    def _require_resample_backend(self) -> Any:
        self._require_available()
        if self._resample_backend is None:
            # This imports the existing V5 CUDA resampler, not a new kernel.
            from tools.openmatte_hdr.cuda_backend import make_cuda_resample_backend

            self._resample_backend = make_cuda_resample_backend()
        return self._resample_backend

    @staticmethod
    def _wrap_device_rgb(payload: Any, source: Frame, *, backend_id: str) -> Frame:
        height, width, channels = (int(value) for value in payload.shape)
        if channels != 3:
            raise ValueError("CUDA preview frame must have three RGB channels")
        return Frame(
            width=width,
            height=height,
            pixel_format="RGB32F",
            bit_depth=32,
            color_range="full",
            primaries="BT.2020",
            transfer="linear",
            matrix="RGB",
            timestamps=source.timestamps,
            memory_domain=MemoryDomain.DEVICE,
            backend=backend_id,
            device=source.device,
            ownership=Ownership.BORROWED,
            memory=FrameMemory(
                payload=payload,
                domain=MemoryDomain.DEVICE,
                nbytes=int(getattr(payload, "nbytes", 0)),
                device=source.device,
                ownership=Ownership.BORROWED,
            ),
            metadata=dict(source.metadata),
        )

    def resize(
        self,
        frame: Frame,
        request: ResizeRequest,
        cancellation: PreviewCancellation,
    ) -> Frame:
        self._require_available()
        cancellation.checkpoint()
        if frame.memory_domain != MemoryDomain.DEVICE:
            raise ValueError("CUDA preview resize requires a device frame")
        if request.memory_domain != MemoryDomain.DEVICE:
            raise ValueError("CUDA preview resize request must target device memory")
        resampler = self._require_resample_backend()
        existing_request = ResampleRequest(
            source_width=request.source_width,
            source_height=request.source_height,
            target_width=request.target_width,
            target_height=request.target_height,
            source_format="RGB32F",
            target_format="RGB32F",
            method=request.method,
        )
        stream = self._cupy.cuda.get_current_stream()
        payload = resampler.process(frame.view(), existing_request, stream)
        cancellation.checkpoint()
        return self._wrap_device_rgb(payload, frame, backend_id=self.backend_id)

    def transform(
        self,
        frame: Frame,
        shot_transform: ShotTransform,
        transform_config: TransformConfig,
        cancellation: PreviewCancellation,
    ) -> Frame:
        self._require_available()
        cancellation.checkpoint()
        if frame.memory_domain != MemoryDomain.DEVICE:
            raise ValueError("CUDA preview transform requires a device frame")
        transform_config.validate()
        backend = self._require_transform_backend()
        workspace = backend.prepare_shot(
            shot_transform,
            sdr_transfer=transform_config.sdr_transfer,
            hdr_transfer=transform_config.hdr_transfer,
            peak_nits=transform_config.peak_nits,
        )
        # transform_device is an adapter boundary over the existing GPU math;
        # it does not duplicate or alter the production equations.
        payload = backend.transform_device(
            frame.view(),
            shot_transform,
            workspace=workspace,
            sdr_transfer=transform_config.sdr_transfer,
            hdr_transfer=transform_config.hdr_transfer,
            peak_nits=transform_config.peak_nits,
        )
        cancellation.checkpoint()
        return self._wrap_device_rgb(payload, frame, backend_id=self.backend_id)

    def materialize(
        self,
        frame: Frame,
        request: PreviewRequest,
        source_role: PreviewMode,
        cancellation: PreviewCancellation,
    ) -> PreviewFrame:
        self._require_available()
        cancellation.checkpoint()
        if frame.memory_domain != MemoryDomain.DEVICE:
            raise ValueError("CUDA preview materialization requires a device frame")
        from numpy import asarray, clip, float32, rint, uint8

        payload = self._cupy.asnumpy(frame.view())
        payload = asarray(payload, dtype=float32)
        if payload.shape != (request.resize.target_height, request.resize.target_width, 3):
            raise ValueError("GPU materialized frame does not match preview request")
        rgb24 = rint(clip(payload, 0.0, 1.0) * 255.0).astype(uint8).tobytes()
        cancellation.checkpoint()
        return PreviewFrame(
            rgb24=rgb24,
            width=request.resize.target_width,
            height=request.resize.target_height,
            frame_number=request.frame_number,
            om_frame_number=request.om_frame_number,
            shot_id=request.shot_id,
            locked_offset=request.locked_offset,
            sync_version=request.sync_version,
            quality=request.quality,
            source_role=source_role,
            renderer_id="single-frame-renderer-v1",
            model_version=request.model_version,
            backend_id=self.backend_id,
            metadata={
                "memory_path": "device -> small host RGB24 display boundary",
                "source_frame_backend": frame.backend,
                "source_frame_metadata": dict(frame.metadata),
            },
        )

    def close(self) -> None:
        transform = self._transform_backend
        if transform is not None and hasattr(transform, "clear"):
            transform.clear()
        self._transform_backend = None
        self._resample_backend = None


class CudaSingleSourceDecoderBackend(SingleSourceDecoderBackend):
    """Single-stream adapter over the existing V5 ``NativeDecoder``."""

    def __init__(
        self,
        *,
        source_mode: int,
        ffmpeg_bin: Path,
        bridge_path: Path,
        device_index: int = 0,
        ring_size: int = 4,
    ) -> None:
        self.source_mode = int(source_mode)
        self.ffmpeg_bin = Path(ffmpeg_bin)
        self.bridge_path = Path(bridge_path)
        self.device_index = int(device_index)
        self.ring_size = int(ring_size)
        self._decoder: Any = None
        self._source: Path | None = None
        self._decoder_name = ""
        self._fps: tuple[int, int] | None = None
        self._next_frame = 0
        self._cancelled = False

    @property
    def lifecycle(self) -> dict[str, Any]:
        native = self._decoder
        return {
            "opened": native is not None,
            "source": str(self._source) if self._source is not None else None,
            "decoder": self._decoder_name,
            "next_frame": int(self._next_frame),
            "native": dict(native.lifecycle) if native is not None else None,
        }

    @property
    def status(self) -> dict[str, Any]:
        return {
            "backend": "cuda-v05-single-source",
            "source_mode": "HDR" if self.source_mode == 0 else "OM",
            "cancelled": self._cancelled,
            "lifecycle": self.lifecycle,
        }

    def open(
        self,
        source: Path,
        decoder: str,
        start_frame: int,
        fps: tuple[int, int] | str | float,
    ) -> None:
        self.close()
        self._source = Path(source)
        self._decoder_name = str(decoder)
        self._fps = CudaPreviewBackend._fps_fraction(fps)
        self._next_frame = int(start_frame)
        self._cancelled = False
        try:
            native_module = CudaPreviewBackend._import_native_module()
            self._decoder = native_module.NativeDecoder(
                self._source,
                self._decoder_name,
                self.source_mode,
                self._next_frame,
                self._fps[0],
                self._fps[1],
                ffmpeg_bin=self.ffmpeg_bin,
                bridge_path=self.bridge_path,
                device_index=self.device_index,
                ring_size=self.ring_size,
            )
        except BaseException:
            self._decoder = None
            raise

    def _require_open(self) -> Any:
        if self._decoder is None or self._source is None or self._fps is None:
            raise PreviewBackendUnavailable("CUDA single-source decoder is not open")
        if self._cancelled:
            raise PreviewCancelled("CUDA single-source decoder was cancelled")
        return self._decoder

    def read(self, frame_number: int) -> Frame | None:
        native = self._require_open()
        requested = int(frame_number)
        if requested < 0:
            raise ValueError("frame_number must be non-negative")
        if requested != self._next_frame:
            self.seek(requested)
            native = self._require_open()
        native_frame = native.next()
        if native_frame is None:
            return None
        try:
            facade = importlib.import_module("tools.openmatte_hdr.cuda_backend")
            frame = facade._frame_from_native(
                native_frame,
                is_hdr=self.source_mode == 0,
                source=SimpleNamespace(
                    device_index=self.device_index,
                    config=SimpleNamespace(fps_text=f"{self._fps[0]}/{self._fps[1]}"),
                ),
            )
        except BaseException:
            native_frame.release()
            raise
        self._next_frame = requested + 1
        return frame

    def seek(self, frame_number: int) -> None:
        if self._source is None or self._fps is None:
            raise PreviewBackendUnavailable("CUDA single-source decoder is not configured")
        requested = int(frame_number)
        if requested < 0:
            raise ValueError("frame_number must be non-negative")
        source, decoder, fps = self._source, self._decoder_name, self._fps
        self.close()
        self.open(source, decoder, requested, fps)

    def cancel(self) -> None:
        self._cancelled = True

    def close(self) -> None:
        native = self._decoder
        self._decoder = None
        if native is not None:
            native.close()


class CudaSingleSourceDecoderSession(PreviewDecoderSession):
    """Preview session binding one configured CUDA single-source backend."""

    def __init__(
        self,
        backend: CudaSingleSourceDecoderBackend,
        *,
        source_path: Path,
        decoder: str,
        start_frame: int,
        fps: tuple[int, int],
        source_role: PreviewMode,
    ) -> None:
        self.backend = backend
        self.source_path = Path(source_path)
        self.decoder_name = str(decoder)
        self.start_frame = int(start_frame)
        self.fps = fps
        self.source_role = source_role

    @property
    def status(self) -> dict[str, Any]:
        result = dict(self.backend.status)
        result["source_role"] = self.source_role.value
        return result

    def open(self) -> None:
        state = self.backend.status
        if not state["lifecycle"]["opened"] or state["cancelled"]:
            self.backend.open(
                self.source_path,
                self.decoder_name,
                self.start_frame,
                self.fps,
            )

    def seek(self, frame_number: int, cancellation: PreviewCancellation) -> None:
        cancellation.checkpoint()
        self.open()
        self.backend.seek(int(frame_number))
        cancellation.checkpoint()

    def read(self, frame_number: int, cancellation: PreviewCancellation) -> Frame:
        cancellation.checkpoint()
        self.open()
        frame = self.backend.read(int(frame_number))
        if frame is None:
            raise PreviewBackendUnavailable(
                f"CUDA decoder ended before {self.source_role.value} frame {frame_number}"
            )
        try:
            cancellation.checkpoint()
        except BaseException:
            frame.release()
            raise
        return frame

    def cancel(self) -> None:
        self.backend.cancel()

    def close(self) -> None:
        self.backend.close()


class CudaDecoderSession(PreviewDecoderSession):
    """Persistent neutral wrapper for an injected single-source CUDA decoder.

    The existing V5 checkout exposes a paired ``CudaDecoderBackend`` rather
    than a single-source seekable decoder.  This wrapper deliberately expects
    a profile-backed single-source adapter instead of pretending the paired
    object is two independent sessions.
    """

    def __init__(self, decoder: Any, *, source_role: PreviewMode, cache_size: int = 2):
        self.decoder = decoder
        self.source_role = source_role
        self.cache_size = int(cache_size)
        self._opened = False
        self._next_frame = 0
        self._cancelled = False

    @property
    def status(self) -> dict[str, Any]:
        return {
            "opened": self._opened,
            "next_frame": self._next_frame,
            "source_role": self.source_role.value,
            "queue_capacity": self.cache_size,
            "decoder": type(self.decoder).__name__,
        }

    def open(self) -> None:
        if self._opened:
            return
        self.decoder.open()
        self._opened = True
        self._next_frame = 0
        self._cancelled = False

    def seek(self, frame_number: int, cancellation: PreviewCancellation) -> None:
        self.open()
        cancellation.checkpoint()
        if hasattr(self.decoder, "seek"):
            self.decoder.seek(int(frame_number))
            self._next_frame = int(frame_number)
            return
        # Re-open is the only honest fallback when the injected decoder has no
        # seek API; it remains a persistent sequential session between seeks.
        self.close()
        self.open()
        for _ in range(int(frame_number)):
            cancellation.checkpoint()
            skipped = self.decoder.read()
            if skipped is None:
                raise PreviewBackendUnavailable("CUDA decoder ended during seek")
            skipped.release()
        self._next_frame = int(frame_number)

    def read(self, frame_number: int, cancellation: PreviewCancellation) -> Frame:
        self.open()
        cancellation.checkpoint()
        if int(frame_number) != self._next_frame:
            self.seek(int(frame_number), cancellation)
        frame = self.decoder.read()
        if frame is None:
            raise PreviewBackendUnavailable("CUDA decoder ended before requested frame")
        if not isinstance(frame, Frame):
            raise PreviewBackendUnavailable(
                "injected CUDA decoder must return a neutral Frame; "
                "paired native batches require a separate coordinator"
            )
        self._next_frame += 1
        return frame

    def cancel(self) -> None:
        self._cancelled = True
        if hasattr(self.decoder, "cancel"):
            self.decoder.cancel()

    def close(self) -> None:
        if self._opened:
            try:
                self.decoder.close()
            finally:
                self._opened = False
