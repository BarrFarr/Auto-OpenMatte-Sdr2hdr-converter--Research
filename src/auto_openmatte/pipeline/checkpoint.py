"""Durable, sidecar render checkpoint primitives.

The checkpoint is deliberately separate from the canonical analysis project.
It records only render orchestration state; image fitting, compositing, sync,
and CUDA implementations remain owned by their existing modules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base error for an invalid or unusable render checkpoint."""


class CheckpointMismatch(CheckpointError):
    """Raised when a checkpoint does not describe the requested video job."""


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def renderer_build_identifier() -> str:
    render_module = Path(__file__).with_name("render.py")
    try:
        return f"canonical-render-segment-v1:{file_sha256(render_module)[:24]}"
    except OSError:
        return "canonical-render-segment-v1:unavailable"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically and durably in the target directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Cannot read render checkpoint {path}: {exc}") from exc
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatch(
            f"Unsupported render checkpoint schema: {payload.get('schema_version')}"
        )
    return payload


def frame_plan(start_frame: int, end_frame: int, segment_frames: int) -> list[dict[str, int]]:
    if end_frame <= start_frame:
        raise CheckpointError("Render range is empty")
    if segment_frames <= 0:
        raise CheckpointError("segment_frames must be positive")
    plans: list[dict[str, int]] = []
    index = 0
    current = int(start_frame)
    while current < end_frame:
        end = min(int(end_frame), current + int(segment_frames))
        plans.append(
            {
                "index": index,
                "frame_start": current,
                "frame_end": end,
                "frame_count": end - current,
            }
        )
        current = end
        index += 1
    return plans


def validate_segment(
    path: Path,
    expected_frames: int,
    *,
    ffprobe_path: Path | None = None,
) -> dict[str, Any] | None:
    """Validate a video-only file and, when available, its video frame count."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    frame_count: int | None = None
    if ffprobe_path is not None:
        import subprocess

        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames,nb_frames",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return None
        try:
            streams = json.loads(result.stdout).get("streams", [])
            stream = streams[0] if streams else {}
            raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
            frame_count = int(raw_count) if raw_count not in (None, "N/A") else None
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        audio_result = subprocess.run(
            [
                str(ffprobe_path),
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if audio_result.returncode != 0:
            return None
        try:
            if json.loads(audio_result.stdout).get("streams", []):
                return None
        except (TypeError, json.JSONDecodeError):
            return None
        if frame_count is not None and frame_count != int(expected_frames):
            return None
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "frame_count": frame_count if frame_count is not None else int(expected_frames),
        "audio_free": True,
    }


def source_ids(project: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("hdr_source", "openmatte_source"):
        source = getattr(project, name, None)
        result[name] = file_signature(getattr(source, "path", Path(""))) if source else None
    return result


def identity_bundle(
    project: Any,
    config: Any,
    output_path: Path,
    *,
    segment_frames: int,
    audio_selection: dict[str, Any],
) -> dict[str, Any]:
    sources = source_ids(project)
    project_identity = {
        "version": getattr(project, "version", "1.0"),
        "sources": sources,
        "sync_model": getattr(project, "sync_model", None),
        "global_geometry": getattr(project, "global_geometry", None),
        "shots": getattr(project, "shots", []),
        "transforms": getattr(project, "transforms", []),
        "transform_config": getattr(project, "transform_config", None),
        "processing_mode": getattr(project, "processing_mode", None),
    }
    project_id = stable_hash(project_identity)
    video_config = {
        "codec": getattr(config, "codec", ""),
        "crf": getattr(config, "crf", 0),
        "pix_fmt": getattr(config, "pix_fmt", ""),
        "resolution": getattr(config, "resolution", ""),
        "encoder_params": getattr(config, "encoder_params", {}),
        "processing_mode": getattr(config, "processing_mode", None),
        "segment_frames": int(segment_frames),
        "output_suffix": Path(output_path).suffix.lower(),
    }
    fitting_identity = {
        "transforms": getattr(project, "transforms", []),
        "transform_config": getattr(project, "transform_config", None),
        "policy": "canonical-fit-missing-transforms-on-resume",
    }
    return {
        "project_id": project_id,
        "source_ids": sources,
        "video_configuration": video_config,
        "video_configuration_hash": stable_hash(video_config),
        "audio_selection": _jsonable(audio_selection),
        "audio_selection_hash": stable_hash(audio_selection),
        "renderer_build_identifier": renderer_build_identifier(),
        "fitting_model_identifier": stable_hash(fitting_identity),
    }


def new_checkpoint(
    identity: dict[str, Any],
    *,
    output_path: Path,
    mode: str,
    start_frame: int,
    end_frame: int,
    segment_frames: int,
    plans: Iterable[dict[str, int]],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "project_id": identity["project_id"],
        "source_ids": identity["source_ids"],
        "profile_ids": [],
        "render_configuration_hash": identity["video_configuration_hash"],
        "video_configuration_hash": identity["video_configuration_hash"],
        "renderer_build_identifier": identity["renderer_build_identifier"],
        "fitting_model_identifier": identity["fitting_model_identifier"],
        "mode": mode,
        "output_path": str(Path(output_path).resolve()),
        "output_video_parameters": identity["video_configuration"],
        "audio_selection": identity["audio_selection"],
        "audio_selection_hash": identity["audio_selection_hash"],
        "segment_plan": {
            "frame_start": int(start_frame),
            "frame_end": int(end_frame),
            "segment_frames": int(segment_frames),
            "segments": list(plans),
        },
        "completed_segments": [],
        "current_segment": None,
        "last_committed_frame": int(start_frame) - 1,
        "timestamp": utc_now(),
        "state": "RUNNING",
        "error": "",
        "final": {"video": None, "mux": None},
    }


def checkpoint_path_for(output_path: Path, configured: str = "", project_path: str | None = None) -> Path:
    if configured:
        return Path(configured).expanduser()
    if project_path:
        return Path(project_path).expanduser().resolve().with_name("render_state.json")
    return Path(output_path).expanduser().resolve().with_name("render_state.json")
