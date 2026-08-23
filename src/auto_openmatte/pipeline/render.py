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
from auto_openmatte.utils.ffmpeg import get_media_tool_config

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


class RenderCancelled(RenderError):
    """Raised when a caller requests cancellation during a render."""


class _RawVideoReader:
    """Small sequential RGB48 reader used by the common output path."""

    def __init__(self, path: Path, width: int, height: int, stream_index: int):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3 * 2
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
        """Read one complete RGB frame, or return None at end of stream."""
        if self._process.stdout is None:
            return None
        raw = bytearray()
        while len(raw) < self._frame_bytes:
            chunk = self._process.stdout.read(self._frame_bytes - len(raw))
            if not chunk:
                return None
            raw.extend(chunk)
        array = np.frombuffer(raw, dtype=np.uint16)
        return array.reshape(self.height, self.width, 3).astype(np.float64) / 65535.0

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
        data = np.rint(np.clip(frame, 0.0, 1.0) * 65535.0).astype(np.uint16).tobytes()
        if len(data) != self._frame_bytes:
            raise RenderError("Encoded frame has an unexpected byte size")
        try:
            self._process.stdin.write(data)
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
    if not project.ready_for_render:
        raise RenderError("Project is not ready for render. Run analysis first.")
    if not project.hdr_source or not project.openmatte_source:
        raise RenderError("Project missing source information.")
    if not project.sync_model.frame_locked:
        raise RenderError("Synchronization is not frame-locked. Cannot render.")

    config.processing_mode = ProcessingMode.coerce(
        forced_mode or project.processing_mode
    )
    transforms = _fit_missing_transforms(project, config, cancel_callback)
    hdr_width, hdr_height, hdr_count, fps = _stream_dimensions(project.hdr_source, "HDR")
    om_width, om_height, om_count, _ = _stream_dimensions(project.openmatte_source, "Open Matte")
    hdr_stream = project.hdr_source.selected_stream
    om_stream = project.openmatte_source.selected_stream
    assert hdr_stream is not None and om_stream is not None

    offset = int(project.sync_model.frame_offset)
    full_start_frame = max(0, -offset)
    full_end_frame = min(hdr_count, om_count - offset)
    if full_end_frame <= full_start_frame:
        raise RenderError("Synchronization offset leaves no overlapping frame range")
    start_frame = full_start_frame if frame_start is None else max(
        full_start_frame, int(frame_start)
    )
    end_frame = full_end_frame if frame_end is None else min(
        full_end_frame, int(frame_end)
    )
    if end_frame <= start_frame:
        raise RenderError("Requested render segment is outside the synchronized frame range")
    total_frames = end_frame - start_frame
    output_width, output_height = _parse_resolution(
        config.resolution,
        (om_width, om_height),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Render %s: %s frames at %sx%s -> %s",
        ProcessingMode.coerce(forced_mode or project.processing_mode).value,
        total_frames,
        output_width,
        output_height,
        output_path,
    )
    if progress_callback:
        progress_callback(0, total_frames)

    hdr_reader = _RawVideoReader(
        project.hdr_source.path,
        hdr_width,
        hdr_height,
        hdr_stream.index,
    )
    om_reader = _RawVideoReader(
        project.openmatte_source.path,
        om_width,
        om_height,
        om_stream.index,
    )
    encoder: _RawVideoEncoder | None = None
    committed = False
    try:
        hdr_reader.discard(start_frame)
        om_reader.discard(start_frame + offset)
        encoder = _RawVideoEncoder(
            output_path,
            output_width,
            output_height,
            fps,
            config,
        )
        convert_renderer = ConvertRenderer()
        convert_config = ConvertOutputConfig(transform=config.transform)
        extend_luts: dict[str, dict] = {}

        for output_index, hdr_frame_index in enumerate(range(start_frame, end_frame), start=1):
            if cancel_callback and cancel_callback():
                raise RenderCancelled("Render cancelled by user")
            hdr_frame = hdr_reader.read()
            om_frame = om_reader.read()
            if hdr_frame is None or om_frame is None:
                raise RenderError("Decoder ended before the synchronized frame range")

            shot = _find_shot_for_frame(project.shots, hdr_frame_index)
            if shot is None:
                raise RenderError(f"No shot found for HDR frame {hdr_frame_index}")
            transform = transforms.get(shot.shot_id)
            if transform is None:
                raise RenderError(f"No transform found for shot {shot.shot_id}")
            mode = _mode_for_shot(project, shot, forced_mode)

            if mode == ProcessingMode.CONVERT:
                output_frame = convert_renderer.apply(om_frame, transform, convert_config)
            else:
                geometry = transform.geometry or project.global_geometry
                mask = _extension_mask(om_frame.shape[:2], geometry)
                lut_key = repr(transform.luminance_curve)
                lut = extend_luts.get(lut_key)
                if lut is None and transform.luminance_curve:
                    lut = build_curve_lut(transform.luminance_curve)
                    extend_luts[lut_key] = lut
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
                )

            output_frame = _resize_frame(output_frame, output_width, output_height)
            encoder.write(output_frame)
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
        hdr_reader.close()
        om_reader.close()
        if encoder is not None and not committed:
            encoder.abort()


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


def render_project_segment(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
    *,
    frame_start: int,
    frame_end: int,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    """Render one closed video-only segment through the canonical renderer.

    This is an orchestration boundary for checkpointed jobs.  It uses the same
    fitting, transform, compositing, reader and encoder implementation as
    ``render_project`` and only restricts the synchronized frame range.
    """
    if config is None:
        config = RenderConfig(processing_mode=project.processing_mode)
    mode = ProcessingMode.coerce(project.processing_mode)
    forced_mode = ProcessingMode.CONVERT if mode == ProcessingMode.CONVERT else ProcessingMode.EXTEND
    return _render_video(
        project,
        Path(output_path),
        config,
        forced_mode=forced_mode,
        frame_start=int(frame_start),
        frame_end=int(frame_end),
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
