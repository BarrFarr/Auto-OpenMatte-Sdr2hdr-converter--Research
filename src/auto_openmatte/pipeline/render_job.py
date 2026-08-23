"""Segmented GUI render orchestration with durable pause/resume state.

This layer owns job lifecycle only.  Each segment is rendered through the
existing canonical renderer and is committed only after close + ffprobe
validation.  It never changes fitting, sync, transform, or CUDA code.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from auto_openmatte.core.config import RenderConfig
from auto_openmatte.core.exceptions import RenderError
from auto_openmatte.core.mode import ProcessingMode
from auto_openmatte.output.mux import concat_video_segments, mux_audio
from auto_openmatte.pipeline.checkpoint import (
    CheckpointMismatch,
    atomic_write_json,
    checkpoint_path_for,
    file_sha256,
    frame_plan,
    identity_bundle,
    load_checkpoint,
    new_checkpoint,
    utc_now,
    validate_segment,
)
from auto_openmatte.pipeline.render import (
    RenderCancelled,
    render_project_segment,
)
from auto_openmatte.utils.ffmpeg import get_media_tool_config

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


class RenderJobCancelled(RenderError):
    """The user cancelled a segmented render."""


class RenderJobRunner:
    """Run/reconcile a segmented canonical render job."""

    def __init__(
        self,
        project: Any,
        output_path: Path,
        config: RenderConfig,
        *,
        project_path: str | None = None,
        checkpoint_path: Path | None = None,
        audio_selection: dict[str, Any] | None = None,
        resume: bool = False,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.project = project
        self.output_path = Path(output_path).expanduser().resolve()
        self.config = config
        self.project_path = project_path
        self.audio_selection = dict(audio_selection or {"type": "NONE"})
        self.resume = bool(resume)
        self.cancel_event = cancel_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.progress_callback = progress_callback
        self.checkpoint_path = checkpoint_path or checkpoint_path_for(
            self.output_path,
            getattr(config, "checkpoint_path", ""),
            project_path,
        )
        self.checkpoint_path = Path(self.checkpoint_path).expanduser().resolve()

    def _source_counts(self) -> tuple[int, int, int, int, float]:
        hdr = getattr(self.project, "hdr_source", None)
        om = getattr(self.project, "openmatte_source", None)
        hdr_stream = getattr(hdr, "selected_stream", None)
        om_stream = getattr(om, "selected_stream", None)
        if hdr_stream is None or om_stream is None:
            raise RenderError("Project has no selected HDR/Open Matte video streams")
        hdr_count = int(hdr_stream.frame_count or round(hdr_stream.duration_seconds * hdr_stream.fps))
        om_count = int(om_stream.frame_count or round(om_stream.duration_seconds * om_stream.fps))
        fps = float(hdr_stream.fps or 0.0)
        if hdr_count <= 0 or om_count <= 0 or fps <= 0:
            raise RenderError("Project sources have no usable frame count/FPS")
        offset = int(self.project.sync_model.frame_offset)
        start = max(0, -offset)
        end = min(hdr_count, om_count - offset)
        if end <= start:
            raise RenderError("Synchronization offset leaves no renderable frame range")
        return start, end, hdr_count, om_count, fps

    def _emit(self, current: int, total: int, **meta: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(current, total, meta)

    def _audio_source(self) -> Path | None:
        source_type = str(self.audio_selection.get("type", "NONE")).upper()
        if source_type == "HDR":
            source = getattr(self.project, "hdr_source", None)
        elif source_type == "OM":
            source = getattr(self.project, "openmatte_source", None)
        else:
            return None
        path = getattr(source, "path", None)
        if path is None or not Path(path).is_file():
            raise RenderError(f"Selected audio source is unavailable: {path}")
        return Path(path)

    def _video_only_path(self) -> Path:
        """Return the persistent merged video artifact used by final muxing."""
        return self.output_path.with_name(
            f".{self.output_path.stem}.video{self.output_path.suffix}"
        )

    def _existing_video_only(
        self,
        state: dict[str, Any],
        expected_frames: int,
    ) -> Path | None:
        """Reuse a checkpointed video merge when only audio selection changed."""
        record = (state.get("final") or {}).get("video") or {}
        raw_path = record.get("path")
        if not raw_path:
            return None
        path = Path(raw_path)
        try:
            ffprobe = get_media_tool_config().require("ffprobe")
        except FileNotFoundError:
            return None
        checked = validate_segment(path, expected_frames, ffprobe_path=ffprobe)
        if checked is None:
            return None
        recorded_hash = str(record.get("sha256") or "")
        if recorded_hash and recorded_hash != checked["sha256"]:
            return None
        return path

    def _initial_or_existing_state(
        self,
        identity: dict[str, Any],
        *,
        start: int,
        end: int,
        segment_frames: int,
        plans: list[dict[str, int]],
    ) -> dict[str, Any]:
        if self.checkpoint_path.is_file():
            if not self.resume:
                raise CheckpointMismatch(
                    f"Render checkpoint exists at {self.checkpoint_path}; use Resume "
                    "or choose a new output/checkpoint path"
                )
            state = load_checkpoint(self.checkpoint_path)
            if state.get("project_id") != identity["project_id"]:
                raise CheckpointMismatch("Render checkpoint project/source identity does not match")
            if state.get("video_configuration_hash") != identity["video_configuration_hash"]:
                raise CheckpointMismatch("Render checkpoint video configuration does not match")
            if state.get("renderer_build_identifier") != identity["renderer_build_identifier"]:
                raise CheckpointMismatch("Render checkpoint renderer build does not match")
            if state.get("fitting_model_identifier") != identity["fitting_model_identifier"]:
                raise CheckpointMismatch("Render checkpoint fitting identity does not match")
            if state.get("segment_plan", {}).get("segments") != plans:
                raise CheckpointMismatch("Render checkpoint segment plan does not match")
            if state.get("audio_selection_hash") != identity["audio_selection_hash"]:
                state["audio_selection"] = identity["audio_selection"]
                state["audio_selection_hash"] = identity["audio_selection_hash"]
                state["final"]["mux"] = None
            return state
        state = new_checkpoint(
            identity,
            output_path=self.output_path,
            mode=ProcessingMode.coerce(self.project.processing_mode).value,
            start_frame=start,
            end_frame=end,
            segment_frames=segment_frames,
            plans=plans,
        )
        atomic_write_json(self.checkpoint_path, state)
        return state

    def _reconcile(self, state: dict[str, Any], plans: list[dict[str, int]], segments_dir: Path) -> dict[int, dict[str, Any]]:
        segments_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.checkpoint_path.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for stale in tmp_dir.glob("segment_*.mkv"):
            try:
                stale.unlink()
            except OSError:
                pass
        ffprobe = get_media_tool_config().require("ffprobe")
        valid: dict[int, dict[str, Any]] = {}
        for record in state.get("completed_segments", []):
            index = int(record.get("index", -1))
            if index < 0 or index >= len(plans):
                continue
            path = segments_dir / f"segment_{index:06d}.mkv"
            checked = validate_segment(path, plans[index]["frame_count"], ffprobe_path=ffprobe)
            if checked is None:
                continue
            recorded_hash = str(record.get("sha256") or "")
            if recorded_hash and recorded_hash != checked["sha256"]:
                continue
            valid[index] = {
                **record,
                "index": index,
                "path": str(path),
                "frame_start": plans[index]["frame_start"],
                "frame_end": plans[index]["frame_end"],
                "frame_count": plans[index]["frame_count"],
                **checked,
            }
            state["completed_segments"] = [valid[index] for index in sorted(valid)]
        contiguous_last = plans[0]["frame_start"] - 1
        for plan in plans:
            if plan["index"] not in valid:
                break
            contiguous_last = plan["frame_end"] - 1
        state["last_committed_frame"] = contiguous_last
        return valid

    def _save(self, state: dict[str, Any], status: str | None = None, error: str = "") -> None:
        state["timestamp"] = utc_now()
        if status is not None:
            state["state"] = status
        if error:
            state["error"] = error
        atomic_write_json(self.checkpoint_path, state)

    def run(self) -> bool:
        start, end, _hdr_count, _om_count, _fps = self._source_counts()
        segment_frames = int(getattr(self.config, "segment_frames", 120))
        plans = frame_plan(start, end, segment_frames)
        identity = identity_bundle(
            self.project,
            self.config,
            self.output_path,
            segment_frames=segment_frames,
            audio_selection=self.audio_selection,
        )
        state = self._initial_or_existing_state(
            identity,
            start=start,
            end=end,
            segment_frames=segment_frames,
            plans=plans,
        )
        segments_dir = self.checkpoint_path.parent / "segments"
        tmp_dir = self.checkpoint_path.parent / "tmp"
        valid = self._reconcile(state, plans, segments_dir)
        state["state"] = "RUNNING"
        state["current_segment"] = None
        self._save(state)
        total_frames = end - start
        completed_frames = sum(record["frame_count"] for record in valid.values())
        self._emit(
            completed_frames,
            total_frames,
            state="RUNNING",
            current_segment=None,
            total_segments=len(plans),
            last_committed_frame=state.get("last_committed_frame", start - 1),
            checkpoint_path=str(self.checkpoint_path),
        )

        try:
            for plan in plans:
                index = plan["index"]
                if index in valid:
                    continue
                if self.cancel_event.is_set():
                    raise RenderJobCancelled("Render cancelled before next segment")
                if self.pause_event.is_set():
                    state["current_segment"] = None
                    self._save(state, "PAUSED")
                    self._emit(completed_frames, total_frames, state="PAUSED", current_segment=None, total_segments=len(plans), last_committed_frame=state.get("last_committed_frame", start - 1), checkpoint_path=str(self.checkpoint_path))
                    return False

                state["current_segment"] = plan
                self._save(state, "RUNNING")
                temp_segment = tmp_dir / f"segment_{index:06d}.mkv"
                final_segment = segments_dir / f"segment_{index:06d}.mkv"
                for stale in (temp_segment, final_segment):
                    if stale.exists() and stale == temp_segment:
                        stale.unlink()

                def segment_progress(current: int, _segment_total: int) -> None:
                    self._emit(
                        completed_frames + int(current),
                        total_frames,
                        state="RUNNING",
                        current_segment=index,
                        total_segments=len(plans),
                        last_committed_frame=state.get("last_committed_frame", start - 1),
                        checkpoint_path=str(self.checkpoint_path),
                    )

                render_project_segment(
                    self.project,
                    temp_segment,
                    config=self.config,
                    frame_start=plan["frame_start"],
                    frame_end=plan["frame_end"],
                    progress_callback=segment_progress,
                    cancel_callback=lambda: self.cancel_event.is_set() or self.pause_event.is_set(),
                )
                ffprobe = get_media_tool_config().require("ffprobe")
                checked = validate_segment(temp_segment, plan["frame_count"], ffprobe_path=ffprobe)
                if checked is None:
                    raise RenderError(f"Segment {index} failed validation")
                os.replace(temp_segment, final_segment)
                checked = validate_segment(final_segment, plan["frame_count"], ffprobe_path=ffprobe)
                if checked is None:
                    raise RenderError(f"Segment {index} failed post-commit validation")
                record = {
                    **plan,
                    "path": str(final_segment),
                    **checked,
                }
                valid[index] = record
                state["completed_segments"] = [valid[key] for key in sorted(valid)]
                state["current_segment"] = None
                state["last_committed_frame"] = max(
                    record["frame_end"] - 1 for record in valid.values()
                )
                completed_frames = sum(item["frame_count"] for item in valid.values())
                self._save(state, "RUNNING")
                self._emit(completed_frames, total_frames, state="RUNNING", current_segment=None, total_segments=len(plans), last_committed_frame=state["last_committed_frame"], checkpoint_path=str(self.checkpoint_path))
                if self.pause_event.is_set():
                    self._save(state, "PAUSED")
                    self._emit(completed_frames, total_frames, state="PAUSED", current_segment=None, total_segments=len(plans), last_committed_frame=state["last_committed_frame"], checkpoint_path=str(self.checkpoint_path))
                    return False

            ordered_segments = [valid[index] for index in range(len(plans))]
            segment_paths = [Path(item["path"]) for item in ordered_segments]
            audio_source = self._audio_source()
            has_audio = audio_source is not None
            video_only = self._existing_video_only(state, total_frames)
            if video_only is None:
                video_only = self._video_only_path()
                concat_video_segments(segment_paths, video_only)
                state["final"]["video"] = {
                    "path": str(video_only),
                    "sha256": file_sha256(video_only),
                    "size": video_only.stat().st_size,
                    "state": "COMPLETE",
                    "audio_free": True,
                }
                # Persist the merged video before attempting audio. A failed
                # mux can therefore resume from this artifact without touching
                # completed segments or repeating concat.
                self._save(state, "RUNNING")

            if has_audio:
                mux_audio(
                    video_only,
                    audio_source,
                    int(self.audio_selection.get("ordinal", 0)),
                    self.output_path,
                )
                state["final"]["mux"] = {
                    "path": str(self.output_path),
                    "audio_selection_hash": state["audio_selection_hash"],
                    "codec_policy": "stream-copy",
                    "state": "COMPLETE",
                }
            else:
                temporary_output = self.output_path.with_name(
                    f".{self.output_path.stem}.video-output{self.output_path.suffix}"
                )
                try:
                    shutil.copyfile(video_only, temporary_output)
                    os.replace(temporary_output, self.output_path)
                finally:
                    try:
                        temporary_output.unlink()
                    except FileNotFoundError:
                        pass
                state["final"]["mux"] = {
                    "path": str(self.output_path),
                    "audio_selection_hash": state["audio_selection_hash"],
                    "codec_policy": "none",
                    "state": "COMPLETE",
                }
            state["current_segment"] = None
            self._save(state, "COMPLETE")
            self._emit(total_frames, total_frames, state="COMPLETE", current_segment=None, total_segments=len(plans), last_committed_frame=state["last_committed_frame"], checkpoint_path=str(self.checkpoint_path))
            return True
        except RenderCancelled as exc:
            state["current_segment"] = None
            if self.pause_event.is_set():
                self._save(state, "PAUSED", str(exc))
                self._emit(completed_frames, total_frames, state="PAUSED", current_segment=None, total_segments=len(plans), last_committed_frame=state.get("last_committed_frame", start - 1), checkpoint_path=str(self.checkpoint_path))
                return False
            self._save(state, "CANCELLED", str(exc))
            self._emit(completed_frames, total_frames, state="CANCELLED", current_segment=None, total_segments=len(plans), last_committed_frame=state.get("last_committed_frame", start - 1), checkpoint_path=str(self.checkpoint_path))
            return False
        except RenderJobCancelled as exc:
            state["current_segment"] = None
            self._save(state, "CANCELLED", str(exc))
            self._emit(completed_frames, total_frames, state="CANCELLED", current_segment=None, total_segments=len(plans), last_committed_frame=state.get("last_committed_frame", start - 1), checkpoint_path=str(self.checkpoint_path))
            return False
        except Exception as exc:
            self._save(state, "FAILED", str(exc))
            raise


def run_segmented_render(*args: Any, **kwargs: Any) -> bool:
    return RenderJobRunner(*args, **kwargs).run()
