"""Frame-locked rendering for EXTEND and reference-based CONVERT.

The renderer owns orchestration and media I/O only.  Fitting is performed by
existing overlap-sampling/luminance functions and applying a shot transform is
delegated to the existing composition/color paths.  Both modes share the same
raw-frame decoder and encoder; their only semantic difference is APPLY:
EXTEND composites the HDR reference in the overlap, while CONVERT applies the
shot transform to the complete Open Matte frame.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import ColorConfig, RenderConfig
from auto_openmatte.core.exceptions import RenderError
from auto_openmatte.core.mode import ProcessingMode
from auto_openmatte.core.models import ProjectData, Shot, ShotTransform
from auto_openmatte.pipeline.compose import composite_extend
from auto_openmatte.pipeline.convert import ConvertOutputConfig, ConvertRenderer
from auto_openmatte.processing.luminance import (
    apply_luminance_curve,
    build_curve_lut,
    estimate_luminance_curve,
)
from auto_openmatte.processing.sampling import sample_overlap_luminance
from auto_openmatte.processing.transform_backend import (
    TransformBackend,
    TransformWorkspace,
    create_transform_backend,
)
from auto_openmatte.utils.ffmpeg import get_media_tool_config

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]

_TRUE_FLAGS = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    """Read a boolean opt-in flag from the environment."""
    return os.environ.get(name, "").strip().lower() in _TRUE_FLAGS


class RenderCancelled(RenderError):
    """Raised when a caller requests cancellation during a render."""


class _RawVideoReader:
    """Small sequential RGB48 reader used by the common output path.

    The returned array is a reusable buffer owned by the reader: it stays valid
    until the next :meth:`read` or :meth:`discard` call on the same reader. At
    4K a frame is about 199 MiB as ``float64``, so allocating a fresh array per
    frame costs more than the decode itself. Callers that need to keep a frame
    beyond the next read must copy it.
    """

    def __init__(self, path: Path, width: int, height: int, stream_index: int):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3 * 2
        # Reused across frames: one raw pipe buffer, one signal-domain buffer.
        self._raw = bytearray(self._frame_bytes)
        self._raw_view = memoryview(self._raw)
        self._raw_samples = np.frombuffer(self._raw, dtype=np.uint16).reshape(
            height, width, 3
        )
        self._signal = np.empty((height, width, 3), dtype=np.float64)
        cmd = [
            get_media_tool_config().require("ffmpeg").as_posix(),
            "-v", "error",
            "-nostdin",
            "-i", str(path),
            "-map", f"0:v:{stream_index}",
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RenderError(f"Unable to open decoder for {path}: {exc}") from exc

    def read(self) -> NDArray[np.floating] | None:
        """Read one complete RGB frame, or return None at end of stream.

        Fills the reader's own buffers rather than allocating, and performs the
        same ``uint16`` widening followed by division by ``65535`` as before, so
        the returned values are unchanged.
        """
        stdout = self._process.stdout
        if stdout is None:
            return None
        filled = 0
        while filled < self._frame_bytes:
            got = stdout.readinto(self._raw_view[filled:])
            if not got:
                return None
            filled += got
        # Single pass: widen and divide together instead of writing the whole
        # frame once to widen it and again to scale it.
        np.divide(self._raw_samples, 65535.0, out=self._signal, casting="unsafe")
        return self._signal

    def discard(self, count: int) -> None:
        """Discard a number of leading frames while preserving stream order."""
        for _ in range(max(0, count)):
            if self.read() is None:
                raise RenderError("Input ended before the synchronized frame range")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class _RawVideoEncoder:
    """Common FFmpeg output backend for both processing modes."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float, config: RenderConfig):
        self.output_path = output_path
        self.temp_path = output_path.with_name(
            f".{output_path.stem}.reference-convert.part{output_path.suffix}"
        )
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3 * 2
        # Reused conversion buffers. The float scratch holds the clipped and
        # scaled frame, the sample buffer the quantized output handed to FFmpeg.
        self._scratch = np.empty((height, width, 3), dtype=np.float64)
        self._samples = np.empty((height, width, 3), dtype=np.uint16)
        self._samples_view = memoryview(self._samples).cast("B")
        if self.temp_path.exists():
            self.temp_path.unlink()

        container = output_path.suffix.lower()
        format_name = {".mkv": "matroska", ".webm": "webm", ".mov": "mov"}.get(
            container
        )
        cmd = [
            get_media_tool_config().require("ffmpeg").as_posix(),
            "-v", "error",
            "-nostdin",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.12g}",
            "-i", "pipe:0",
            "-an",
            "-c:v", config.codec,
            "-crf", str(config.crf),
            "-pix_fmt", config.pix_fmt,
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            "-y",
        ]
        if format_name:
            cmd.extend(["-f", format_name])
        for key, value in config.encoder_params.items():
            option = key if key.startswith("-") else f"-{key}"
            cmd.extend([option, str(value)])
        cmd.append(str(self.temp_path))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RenderError(f"Unable to open encoder for {output_path}: {exc}") from exc

    def write(self, frame: NDArray[np.floating]) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise RenderError(
                f"Output frame shape {frame.shape} does not match "
                f"encoder shape {(self.height, self.width, 3)}"
            )
        if self._process.stdin is None:
            raise RenderError("Encoder input is not available")
        # Same clip, scale, round and narrow as before, but through preallocated
        # buffers and without an extra copy for the pipe write.
        np.clip(frame, 0.0, 1.0, out=self._scratch)
        self._scratch *= 65535.0
        np.rint(self._scratch, out=self._scratch)
        np.copyto(self._samples, self._scratch, casting="unsafe")
        if self._samples_view.nbytes != self._frame_bytes:
            raise RenderError("Encoded frame has an unexpected byte size")
        try:
            self._process.stdin.write(self._samples_view)
        except (BrokenPipeError, OSError) as exc:
            raise RenderError("FFmpeg encoder stopped while writing output") from exc

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        return_code = self._process.wait(timeout=120)
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        if return_code != 0:
            message = stderr.decode(errors="replace").strip()
            raise RenderError(f"FFmpeg encoder failed ({return_code}): {message}")

    def abort(self) -> None:
        process = self._process
        if process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            process.kill()
            process.wait(timeout=10)
        if self.temp_path.exists():
            self.temp_path.unlink()

    def commit(self) -> None:
        if not self.temp_path.is_file() or self.temp_path.stat().st_size == 0:
            raise RenderError("FFmpeg completed without producing a non-empty output")
        os.replace(self.temp_path, self.output_path)


def _find_shot_for_frame(shots: list[Shot], frame: int) -> Shot | None:
    """Find which shot a given HDR frame belongs to."""
    for shot in shots:
        if shot.hdr_start_frame <= frame < shot.hdr_end_frame:
            return shot
    if shots:
        return shots[-1]
    return None


def _find_transform_for_shot(
    transforms: list[ShotTransform], shot_id: int
) -> ShotTransform | None:
    """Find a fitted transform for one shot."""
    for transform in transforms:
        if transform.shot_id == shot_id:
            return transform
    return None


def _stream_dimensions(source, label: str) -> tuple[int, int, int, float]:
    stream = source.selected_stream if source else None
    if stream is None or stream.width <= 0 or stream.height <= 0:
        raise RenderError(f"{label} source has no usable selected video stream")
    fps = float(stream.fps or 24.0)
    count = int(stream.frame_count or round(stream.duration_seconds * fps))
    if count <= 0:
        raise RenderError(f"{label} source has no usable frame count")
    return int(stream.width), int(stream.height), count, fps


def _parse_resolution(value: str, fallback: tuple[int, int]) -> tuple[int, int]:
    if not value:
        return fallback
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value, re.IGNORECASE)
    if not match:
        raise RenderError(f"Invalid output resolution {value!r}; expected WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise RenderError("Output resolution must be positive")
    return width, height


def _resize_frame(frame: NDArray[np.floating], width: int, height: int) -> NDArray[np.floating]:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    from scipy.ndimage import zoom

    resized = zoom(
        frame,
        (height / frame.shape[0], width / frame.shape[1], 1.0),
        order=1,
    )
    return resized[:height, :width, :]


def _fit_missing_transforms(
    project: ProjectData,
    config: RenderConfig,
    cancel_callback: CancelCallback | None,
) -> dict[int, ShotTransform]:
    """Fit only missing shots using the existing paired-overlap pipeline."""
    if not project.shots:
        raise RenderError("Project contains no shots for shot-level fitting")
    if project.hdr_source is None or project.openmatte_source is None:
        raise RenderError("Project is missing sources required for overlap fitting")

    transforms = {transform.shot_id: transform for transform in project.transforms}
    color_config = ColorConfig()
    for shot in project.shots:
        if shot.shot_id in transforms:
            continue
        if cancel_callback and cancel_callback():
            raise RenderCancelled("Render cancelled during shot fitting")
        geometry = project.global_geometry
        samples = sample_overlap_luminance(
            project.hdr_source,
            project.openmatte_source,
            project.sync_model,
            geometry,
            shot,
            config=color_config,
            n_samples=color_config.samples_per_shot,
        )
        curve = estimate_luminance_curve(
            samples.sdr_luminance,
            samples.hdr_luminance,
            config=color_config,
        )
        confidence = 0.0
        if samples.n_valid_pairs >= 2:
            mapped = apply_luminance_curve(samples.sdr_luminance, curve)
            if np.std(mapped) > 0 and np.std(samples.hdr_luminance) > 0:
                confidence = max(
                    0.0,
                    float(np.corrcoef(mapped, samples.hdr_luminance)[0, 1]),
                )
        transforms[shot.shot_id] = ShotTransform(
            shot_id=shot.shot_id,
            luminance_curve=curve,
            confidence=confidence,
            geometry=geometry if not geometry.is_global else None,
        )
        logger.info(
            "Shot %s fitted from %s paired overlap samples (confidence %.4f)",
            shot.shot_id,
            samples.n_valid_pairs,
            confidence,
        )
    project.transforms = list(transforms.values())
    config.transform.validate()
    return transforms


def _extension_mask(shape: tuple[int, int], geometry) -> NDArray[np.floating]:
    """Build the existing EXTEND overlap/extension mask from geometry."""
    height, width = shape
    x1, y1, x2, y2 = (int(round(value)) for value in geometry.overlap_bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise RenderError("Geometry has no valid overlap region for EXTEND")
    mask = np.ones((height, width), dtype=np.float64)
    mask[y1:y2, x1:x2] = 0.0
    return mask


def _mode_for_shot(
    project: ProjectData,
    shot: Shot,
    forced_mode: ProcessingMode | None,
) -> ProcessingMode:
    if forced_mode is not None:
        return forced_mode
    return ProcessingMode.coerce(shot.processing_mode or project.processing_mode)


class RenderSession:
    """Holds the decoders and derived parameters for one render job.

    A segmented job renders many consecutive frame ranges. Opening a fresh
    decoder per range would restart both sources at frame zero and step through
    every earlier frame again, which makes the total cost grow with the square
    of the segment count. Keeping one session for the whole job decodes each
    source once.

    Frame order, the pixel conversion, the per-frame shot/transform lookup and
    the composited result are identical to rendering the same range with a fresh
    decoder; only the redundant re-decoding is removed.
    """

    def __init__(
        self,
        project: ProjectData,
        config: RenderConfig,
        *,
        forced_mode: ProcessingMode | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> None:
        if not project.ready_for_render:
            raise RenderError("Project is not ready for render. Run analysis first.")
        if not project.hdr_source or not project.openmatte_source:
            raise RenderError("Project missing source information.")
        if not project.sync_model.frame_locked:
            raise RenderError("Synchronization is not frame-locked. Cannot render.")

        config.processing_mode = ProcessingMode.coerce(
            forced_mode or project.processing_mode
        )
        self.project = project
        self.config = config
        self.forced_mode = forced_mode
        self.transforms = _fit_missing_transforms(project, config, cancel_callback)

        hdr_width, hdr_height, hdr_count, fps = _stream_dimensions(
            project.hdr_source, "HDR"
        )
        om_width, om_height, om_count, _ = _stream_dimensions(
            project.openmatte_source, "Open Matte"
        )
        hdr_stream = project.hdr_source.selected_stream
        om_stream = project.openmatte_source.selected_stream
        assert hdr_stream is not None and om_stream is not None

        self.hdr_width, self.hdr_height = hdr_width, hdr_height
        self.om_width, self.om_height = om_width, om_height
        self.fps = fps
        self.hdr_stream = hdr_stream
        self.om_stream = om_stream

        self.offset = int(project.sync_model.frame_offset)
        self.full_start_frame = max(0, -self.offset)
        self.full_end_frame = min(hdr_count, om_count - self.offset)
        if self.full_end_frame <= self.full_start_frame:
            raise RenderError("Synchronization offset leaves no overlapping frame range")

        self.output_width, self.output_height = _parse_resolution(
            config.resolution,
            (om_width, om_height),
        )
        self._convert_renderer = ConvertRenderer()
        self._convert_config = ConvertOutputConfig(transform=config.transform)
        self._extend_luts: dict[str, dict] = {}

        # Optional GPU transform for the EXTEND extension strips. Opt-in through
        # the environment so no persisted render configuration or checkpoint
        # identity changes. The backend keeps a sticky CPU fallback, so a missing
        # or failing CUDA runtime degrades instead of breaking the render.
        # It is not bit-identical to the CPU backend: measured agreement is
        # within a small fraction of one 16-bit output code.
        self.gpu_transform_requested = _env_flag("OPENMATTE_GPU_TRANSFORM")
        self._transform_backend: TransformBackend | None = None
        self._transform_workspaces: dict[int, TransformWorkspace] = {}
        if self.gpu_transform_requested:
            try:
                self._transform_backend = create_transform_backend(experimental_gpu=True)
                logger.info(
                    "Render session using transform backend %r",
                    getattr(self._transform_backend, "name", "unknown"),
                )
            except Exception as exc:  # noqa: BLE001 - stay on the CPU path
                logger.warning(
                    "GPU transform backend unavailable (%s: %s); using the CPU path",
                    type(exc).__name__,
                    exc,
                )
                self._transform_backend = None

        self._hdr_reader: _RawVideoReader | None = None
        self._om_reader: _RawVideoReader | None = None
        # Index of the next frame each decoder will return, tracked separately
        # because the two streams are offset by the sync model.
        self._next_hdr_frame = 0
        self._next_om_frame = 0
        self.decoder_starts = 0
        self.frames_skipped = 0
        # Both frames of a pair are moved over separate pipes. Transferring them
        # one after the other adds their times together even though the two
        # FFmpeg processes decode concurrently, so the reads are overlapped.
        # The pipe read and the NumPy conversion both release the GIL.
        self._readers_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="om-render-read"
        )

    # -- range helpers ---------------------------------------------------

    def clamp_range(
        self, frame_start: int | None, frame_end: int | None
    ) -> tuple[int, int]:
        """Clamp a requested range to the synchronized overlap."""
        start = (
            self.full_start_frame
            if frame_start is None
            else max(self.full_start_frame, int(frame_start))
        )
        end = (
            self.full_end_frame
            if frame_end is None
            else min(self.full_end_frame, int(frame_end))
        )
        if end <= start:
            raise RenderError(
                "Requested render segment is outside the synchronized frame range"
            )
        return start, end

    # -- decoder lifecycle -----------------------------------------------

    def _open_readers(self) -> None:
        self._close_readers()
        assert self.project.hdr_source is not None
        assert self.project.openmatte_source is not None
        self._hdr_reader = _RawVideoReader(
            self.project.hdr_source.path,
            self.hdr_width,
            self.hdr_height,
            self.hdr_stream.index,
        )
        self._om_reader = _RawVideoReader(
            self.project.openmatte_source.path,
            self.om_width,
            self.om_height,
            self.om_stream.index,
        )
        self._next_hdr_frame = 0
        self._next_om_frame = 0
        self.decoder_starts += 1

    def _close_readers(self) -> None:
        for reader in (self._hdr_reader, self._om_reader):
            if reader is not None:
                reader.close()
        self._hdr_reader = None
        self._om_reader = None

    def position_at(self, hdr_frame: int) -> None:
        """Place both decoders so the next read returns ``hdr_frame``.

        Moving forward steps through the intermediate frames, exactly as a fresh
        decoder would. Moving backward is not possible on a sequential pipe, so
        the decoders are restarted, which costs the same as the previous
        behaviour rather than more.
        """
        target_om_frame = hdr_frame + self.offset
        if self._hdr_reader is None or self._om_reader is None:
            self._open_readers()
        if hdr_frame < self._next_hdr_frame or target_om_frame < self._next_om_frame:
            logger.info(
                "Render session rewinding to frame %s; restarting decoders",
                hdr_frame,
            )
            self._open_readers()
        assert self._hdr_reader is not None and self._om_reader is not None
        hdr_skip = hdr_frame - self._next_hdr_frame
        om_skip = target_om_frame - self._next_om_frame
        pending = []
        if hdr_skip > 0:
            pending.append(self._readers_pool.submit(self._hdr_reader.discard, hdr_skip))
        if om_skip > 0:
            pending.append(self._readers_pool.submit(self._om_reader.discard, om_skip))
        for future in pending:
            future.result()
        if hdr_skip > 0:
            self._next_hdr_frame = hdr_frame
            self.frames_skipped += hdr_skip
        if om_skip > 0:
            self._next_om_frame = target_om_frame
            self.frames_skipped += om_skip

    def read_pair(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Read one synchronized HDR/Open Matte frame pair."""
        if self._hdr_reader is None or self._om_reader is None:
            raise RenderError("Render session decoders are not open")
        hdr_future = self._readers_pool.submit(self._hdr_reader.read)
        om_future = self._readers_pool.submit(self._om_reader.read)
        hdr_frame = hdr_future.result()
        om_frame = om_future.result()
        if hdr_frame is None or om_frame is None:
            raise RenderError("Decoder ended before the synchronized frame range")
        self._next_hdr_frame += 1
        self._next_om_frame += 1
        return hdr_frame, om_frame

    # -- per-frame composition -------------------------------------------

    def compose(
        self,
        hdr_frame: NDArray[np.floating],
        om_frame: NDArray[np.floating],
        hdr_frame_index: int,
    ) -> NDArray[np.floating]:
        """Produce the output frame for one synchronized pair."""
        project = self.project
        config = self.config
        shot = _find_shot_for_frame(project.shots, hdr_frame_index)
        if shot is None:
            raise RenderError(f"No shot found for HDR frame {hdr_frame_index}")
        transform = self.transforms.get(shot.shot_id)
        if transform is None:
            raise RenderError(f"No transform found for shot {shot.shot_id}")
        mode = _mode_for_shot(project, shot, self.forced_mode)

        if mode == ProcessingMode.CONVERT:
            output_frame = self._convert_renderer.apply(
                om_frame, transform, self._convert_config
            )
        else:
            geometry = transform.geometry or project.global_geometry
            mask = _extension_mask(om_frame.shape[:2], geometry)
            lut_key = repr(transform.luminance_curve)
            lut = self._extend_luts.get(lut_key)
            if lut is None and transform.luminance_curve:
                lut = build_curve_lut(transform.luminance_curve)
                self._extend_luts[lut_key] = lut
            output_frame = composite_extend(
                hdr_frame,
                om_frame,
                transform,
                geometry,
                mask,
                sdr_transfer=config.transform.sdr_transfer,
                hdr_transfer=config.transform.hdr_transfer,
                peak_nits=config.transform.peak_nits,
                prebuilt_lut=lut,
                backend=self._transform_backend,
                backend_workspace=self._workspace_for(shot.shot_id, transform),
            )

        return _resize_frame(output_frame, self.output_width, self.output_height)

    def _workspace_for(
        self, shot_id: int, transform: ShotTransform
    ) -> TransformWorkspace | None:
        """Return the per-shot backend workspace, preparing it on first use."""
        backend = self._transform_backend
        if backend is None:
            return None
        workspace = self._transform_workspaces.get(shot_id)
        if workspace is None:
            workspace = backend.prepare_shot(
                transform,
                sdr_transfer=self.config.transform.sdr_transfer,
                hdr_transfer=self.config.transform.hdr_transfer,
                peak_nits=self.config.transform.peak_nits,
            )
            self._transform_workspaces[shot_id] = workspace
        return workspace

    def close(self) -> None:
        self._close_readers()
        self._readers_pool.shutdown(wait=True)
        backend = self._transform_backend
        if backend is not None:
            clear = getattr(backend, "clear", None)
            if clear is not None:
                clear()
        self._transform_workspaces.clear()

    def __enter__(self) -> RenderSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def render_segment_with_session(
    session: RenderSession,
    output_path: Path,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    """Render one closed segment from an already open :class:`RenderSession`."""
    start_frame, end_frame = session.clamp_range(frame_start, frame_end)
    total_frames = end_frame - start_frame
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Render %s: %s frames at %sx%s -> %s",
        ProcessingMode.coerce(
            session.forced_mode or session.project.processing_mode
        ).value,
        total_frames,
        session.output_width,
        session.output_height,
        output_path,
    )
    if progress_callback:
        progress_callback(0, total_frames)

    session.position_at(start_frame)
    encoder = _RawVideoEncoder(
        output_path,
        session.output_width,
        session.output_height,
        session.fps,
        session.config,
    )
    committed = False
    try:
        for output_index, hdr_frame_index in enumerate(
            range(start_frame, end_frame), start=1
        ):
            if cancel_callback and cancel_callback():
                raise RenderCancelled("Render cancelled by user")
            hdr_frame, om_frame = session.read_pair()
            encoder.write(session.compose(hdr_frame, om_frame, hdr_frame_index))
            if progress_callback:
                progress_callback(output_index, total_frames)
        encoder.close()
        encoder.commit()
        committed = True
        return True
    except RenderCancelled:
        raise
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"Render failed: {exc}") from exc
    finally:
        if not committed:
            encoder.abort()


def _render_video(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig,
    *,
    forced_mode: ProcessingMode | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    with RenderSession(
        project,
        config,
        forced_mode=forced_mode,
        cancel_callback=cancel_callback,
    ) as session:
        return render_segment_with_session(
            session,
            output_path,
            frame_start=frame_start,
            frame_end=frame_end,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )


def render_extend(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    """Render EXTEND: HDR reference center plus seam-aware OM extensions."""
    if config is None:
        config = RenderConfig(processing_mode=ProcessingMode.EXTEND)
    logger.info("Render Mode EXTEND: HDR center + Open Matte extensions")
    return _render_video(
        project,
        output_path,
        config,
        forced_mode=ProcessingMode.EXTEND,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def render_convert_hdr(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    """Render CONVERT: full-frame OM SDR-to-HDR using HDR only for fitting."""
    if config is None:
        config = RenderConfig(processing_mode=ProcessingMode.CONVERT)
    logger.info("Render Mode CONVERT: full-frame reference-based SDR -> HDR")
    return _render_video(
        project,
        output_path,
        config,
        forced_mode=ProcessingMode.CONVERT,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def render_project(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    """Render using the mode stored in the project/shot configuration."""
    if config is None:
        config = RenderConfig(processing_mode=project.processing_mode)
    mode = ProcessingMode.coerce(project.processing_mode)
    if mode == ProcessingMode.CONVERT:
        return render_convert_hdr(
            project,
            output_path,
            config=config,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    return render_extend(
        project,
        output_path,
        config=config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def segment_mode_for(project: ProjectData) -> ProcessingMode:
    """Return the forced mode a segmented job renders with."""
    mode = ProcessingMode.coerce(project.processing_mode)
    return ProcessingMode.CONVERT if mode == ProcessingMode.CONVERT else ProcessingMode.EXTEND


def render_project_segment(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
    *,
    frame_start: int,
    frame_end: int,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    session: RenderSession | None = None,
) -> bool:
    """Render one closed video-only segment through the canonical renderer.

    This is an orchestration boundary for checkpointed jobs.  It uses the same
    fitting, transform, compositing, reader and encoder implementation as
    ``render_project`` and only restricts the synchronized frame range.

    Passing an open ``session`` lets consecutive segments share one pair of
    decoders instead of restarting both sources from frame zero per segment.
    """
    if config is None:
        config = RenderConfig(processing_mode=project.processing_mode)
    if session is not None:
        return render_segment_with_session(
            session,
            Path(output_path),
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    return _render_video(
        project,
        Path(output_path),
        config,
        forced_mode=segment_mode_for(project),
        frame_start=int(frame_start),
        frame_end=int(frame_end),
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
