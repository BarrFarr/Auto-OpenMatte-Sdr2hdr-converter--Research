"""Final video-only concat and audio mux boundaries.

Video rendering remains audio-free. This module only probes audio streams and
performs the final container operation after all video segments are complete.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from auto_openmatte.utils.ffmpeg import get_media_tool_config


class MediaMuxError(RuntimeError):
    """Raised when concat or final audio muxing fails."""


def probe_audio_tracks(path: Path, *, ffprobe_path: Path | None = None) -> list[dict[str, Any]]:
    config = get_media_tool_config()
    ffprobe = Path(ffprobe_path) if ffprobe_path else config.require("ffprobe")
    result = subprocess.run(
        [
            str(ffprobe),
            "-v", "error",
            "-select_streams", "a",
            "-show_entries",
            "stream=index,codec_name,codec_long_name,channels,channel_layout,sample_rate,disposition:stream_tags=language,title",
            "-of", "json",
            str(Path(path)),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise MediaMuxError(
            f"ffprobe audio failed for {path}: {result.stderr.strip()}"
        )
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise MediaMuxError(f"ffprobe returned invalid audio JSON for {path}") from exc
    tracks: list[dict[str, Any]] = []
    for ordinal, stream in enumerate(streams):
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        codec = str(stream.get("codec_name") or "")
        language = str(tags.get("language") or "")
        title = str(tags.get("title") or "")
        label = " / ".join(value for value in (codec, language, title) if value)
        tracks.append(
            {
                "ordinal": ordinal,
                "global_index": int(stream.get("index", ordinal)),
                "codec_name": codec,
                "codec_long_name": str(stream.get("codec_long_name") or ""),
                "channels": int(stream.get("channels") or 0),
                "channel_layout": str(stream.get("channel_layout") or ""),
                "sample_rate": int(stream.get("sample_rate") or 0),
                "language": language,
                "title": title,
                "default": bool(int(disposition.get("default", 0) or 0)),
                "label": label or f"Audio track {ordinal + 1}",
            }
        )
    return tracks


def _run_ffmpeg(command: list[str], description: str) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise MediaMuxError(f"{description} failed ({result.returncode}): {detail}")


def _assert_video_only(path: Path, ffprobe_path: Path) -> None:
    result = subprocess.run(
        [
            str(ffprobe_path),
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "json",
            str(Path(path)),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise MediaMuxError(
            f"video-only validation failed for {path}: {result.stderr.strip()}"
        )
    try:
        audio_streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise MediaMuxError(f"video-only validation returned invalid JSON for {path}") from exc
    if audio_streams:
        raise MediaMuxError(f"video-only artifact unexpectedly contains audio: {path}")


def _atomic_temp_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f".{output_path.stem}.{suffix}{output_path.suffix}")


def concat_video_segments(
    segment_paths: list[Path],
    output_path: Path,
    *,
    ffmpeg_path: Path | None = None,
) -> Path:
    """Concatenate video-only segments with stream copy and atomic commit."""
    if not segment_paths:
        raise MediaMuxError("Cannot concat an empty segment list")
    config = get_media_tool_config()
    ffmpeg = Path(ffmpeg_path) if ffmpeg_path else config.require("ffmpeg")
    ffprobe = config.require("ffprobe")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_name(f".{output_path.stem}.concat.txt")
    temporary = _atomic_temp_path(output_path, "concat")
    try:
        lines = []
        for segment in segment_paths:
            normalized = str(Path(segment).resolve()).replace("\\", "/")
            lines.append("file '" + normalized.replace("'", "'\\''") + "'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run_ffmpeg(
            [
                str(ffmpeg),
                "-v", "error",
                "-nostdin",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_path),
                "-map", "0:v:0",
                "-c", "copy",
                "-an",
                "-y",
                str(temporary),
            ],
            "video segment concat",
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise MediaMuxError("FFmpeg concat produced no non-empty video")
        _assert_video_only(temporary, ffprobe)
        os.replace(temporary, output_path)
    finally:
        for path in (list_path, temporary):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise MediaMuxError("FFmpeg concat produced no non-empty video")
    return output_path


def mux_audio(
    video_path: Path,
    audio_source_path: Path,
    audio_ordinal: int,
    output_path: Path,
    *,
    ffmpeg_path: Path | None = None,
) -> Path:
    """Mux selected audio at the final boundary with stream copy only."""
    config = get_media_tool_config()
    ffmpeg = Path(ffmpeg_path) if ffmpeg_path else config.require("ffmpeg")
    ffprobe = config.require("ffprobe")
    video_path = Path(video_path)
    audio_source_path = Path(audio_source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_video_only(video_path, ffprobe)
    tracks = probe_audio_tracks(audio_source_path, ffprobe_path=ffprobe)
    if int(audio_ordinal) < 0 or int(audio_ordinal) >= len(tracks):
        raise MediaMuxError(
            f"Selected audio ordinal {audio_ordinal} is unavailable in {audio_source_path}"
        )
    temporary = _atomic_temp_path(output_path, "mux")
    try:
        _run_ffmpeg(
            [
                str(ffmpeg),
                "-v", "error",
                "-nostdin",
                "-i", str(video_path),
                "-i", str(audio_source_path),
                "-map", "0:v:0",
                "-map", f"1:a:{int(audio_ordinal)}",
                "-map_metadata", "0",
                "-c:v", "copy",
                "-c:a", "copy",
                "-y",
                str(temporary),
            ],
            "audio stream-copy mux",
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise MediaMuxError("Audio mux produced no non-empty output")
        if not probe_audio_tracks(temporary, ffprobe_path=ffprobe):
            raise MediaMuxError("Audio mux produced no audio stream")
        os.replace(temporary, output_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise MediaMuxError("Audio mux produced no non-empty output")
    return output_path
