"""Optional fast synchronization strategy limited to the first ten minutes.

The reference synchronizer remains in ``analysis.sync``.  This module is an
independent proposal strategy for the GUI: it extracts only keyframes from the
first ten minutes, ranks evenly distributed representative samples using cheap
image descriptors, votes on offset candidates, and verifies the best candidates
with short consecutive grayscale windows.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import SyncConfig
from auto_openmatte.core.exceptions import SynchronizationError
from auto_openmatte.core.models import SourceInfo, SyncModel, SyncStatus
from auto_openmatte.utils.ffmpeg import get_media_tool_config
from auto_openmatte.utils.frames import extract_segment_at_time_grayscale

logger = logging.getLogger(__name__)

FAST_SYNC_MAX_SECONDS = 10 * 60.0
_FAST_PROXY_WIDTH = 96
_FAST_POINT_COUNT = 8
_FAST_TOP_MATCHES = 5
_FAST_WINDOW_FRAMES = 9
_FAST_VERIFY_ANCHORS = 3
_FAST_VERIFY_WIDTH = 128
_FAST_MIN_SIMILARITY = 0.35
_FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES = 3
_FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES = 1
_FAST_DIAGNOSTIC_MAX_AGREEMENT_TOLERANCE_FRAMES = 1
_FAST_DIAGNOSTIC_MIN_MARGIN = 0.10
_FAST_DIAGNOSTIC_MIN_CONFIDENCE = 0.80
_PROGRESS = Callable[[str], None]


@dataclass
class _FastFeature:
    timestamp: float
    frame_index: int
    luma_hist: NDArray[np.floating]
    chroma_hist: NDArray[np.floating]
    gradient_hist: NDArray[np.floating]
    spatial: NDArray[np.floating]
    edge_energy: float
    luma_std: float
    characteristic_score: float = 0.0


def _video_ordinal(source: SourceInfo) -> int:
    selected = source.selected_stream
    if selected is None:
        return 0
    for ordinal, stream in enumerate(source.video_streams):
        if stream is selected or stream.index == selected.index:
            return ordinal
    return 0


def _source_duration(source: SourceInfo) -> float:
    stream = source.selected_stream
    if stream is None:
        return 0.0
    if stream.duration_seconds > 0:
        return float(stream.duration_seconds)
    fps = float(stream.fps or 0.0)
    count = int(stream.frame_count or 0)
    return count / fps if fps > 0 and count > 0 else 0.0


def _progress(callback: _PROGRESS | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _extract_keyframe_clip(
    source: SourceInfo,
    limit_seconds: float,
    output_path: Path,
) -> None:
    """Write a tiny keyframe-only clip for the bounded analysis interval."""
    ffmpeg = get_media_tool_config().require("ffmpeg")
    ordinal = _video_ordinal(source)
    command = [
        str(ffmpeg),
        "-v", "error",
        "-nostdin",
        "-copyts",
        "-skip_frame", "nokey",
        "-i", str(source.path),
        "-map", f"0:v:{ordinal}",
        "-t", f"{limit_seconds:.3f}",
        "-vf", f"scale={_FAST_PROXY_WIDTH}:-2,format=rgb24",
        "-fps_mode", "passthrough",
        "-an",
        "-c:v", "ffv1",
        "-g", "1",
        "-f", "matroska",
        "-y", str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SynchronizationError(
            f"Fast sync keyframe extraction failed for {source.path}: {exc}"
        ) from exc
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        detail = (result.stderr or result.stdout).strip()
        raise SynchronizationError(
            f"Fast sync could not extract keyframes from {source.path}: {detail}"
        )


def _clip_timestamps(clip_path: Path, fps: float) -> list[float]:
    ffprobe = get_media_tool_config().require("ffprobe")
    result = subprocess.run(
        [
            str(ffprobe),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time",
            "-of", "json",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise SynchronizationError(
            f"Fast sync timestamp probe failed for {clip_path}: {result.stderr.strip()}"
        )
    try:
        frames = json.loads(result.stdout).get("frames", [])
    except json.JSONDecodeError as exc:
        raise SynchronizationError(
            f"Fast sync timestamp probe returned invalid JSON for {clip_path}"
        ) from exc
    timestamps: list[float] = []
    for index, frame in enumerate(frames):
        raw = frame.get("best_effort_timestamp_time")
        if raw in (None, "N/A"):
            raw = frame.get("pkt_pts_time")
        try:
            timestamps.append(float(raw))
        except (TypeError, ValueError):
            timestamps.append(index / fps if fps > 0 else float(index))
    return timestamps


def _decode_clip_rgb(
    clip_path: Path,
    source: SourceInfo,
) -> list[NDArray[np.uint8]]:
    ffmpeg = get_media_tool_config().require("ffmpeg")
    stream = source.selected_stream
    if stream is None or stream.width <= 0 or stream.height <= 0:
        raise SynchronizationError("Fast sync source has no usable dimensions")
    height = max(2, int(round(stream.height * _FAST_PROXY_WIDTH / stream.width)))
    if height % 2:
        height -= 1
    frame_bytes = _FAST_PROXY_WIDTH * height * 3
    result = subprocess.run(
        [
            str(ffmpeg),
            "-v", "error",
            "-nostdin",
            "-i", str(clip_path),
            "-map", "0:v:0",
            "-fps_mode", "passthrough",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise SynchronizationError(
            f"Fast sync thumbnail decode failed for {clip_path}"
        )
    raw = result.stdout
    frames: list[NDArray[np.uint8]] = []
    for offset in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        frame = np.frombuffer(raw[offset:offset + frame_bytes], dtype=np.uint8)
        frames.append(frame.reshape(height, _FAST_PROXY_WIDTH, 3).copy())
    return frames


def _normalise_histogram(
    values: NDArray[np.floating],
    bins: int,
    value_range: tuple[float, float],
) -> NDArray[np.floating]:
    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    histogram = histogram.astype(np.float64)
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def _feature_from_frame(
    frame: NDArray[np.uint8],
    timestamp: float,
    frame_index: int,
) -> _FastFeature:
    rgb = frame.astype(np.float64) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    chroma = np.stack((rgb[..., 0] - rgb[..., 1], rgb[..., 1] - rgb[..., 2]), axis=-1)
    gx = np.diff(luma, axis=1, prepend=luma[:, :1])
    gy = np.diff(luma, axis=0, prepend=luma[:1, :])
    gradient = np.hypot(gx, gy)
    spatial = cv2.resize(
        luma.astype(np.float32),
        (16, 9),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)
    spatial -= spatial.mean()
    spatial_std = float(spatial.std())
    if spatial_std > 1e-8:
        spatial /= spatial_std
    return _FastFeature(
        timestamp=float(timestamp),
        frame_index=int(frame_index),
        luma_hist=_normalise_histogram(luma, 16, (0.0, 1.0)),
        chroma_hist=np.concatenate(
            (
                _normalise_histogram(chroma[..., 0], 8, (-1.0, 1.0)),
                _normalise_histogram(chroma[..., 1], 8, (-1.0, 1.0)),
            )
        ),
        gradient_hist=_normalise_histogram(np.clip(gradient * 4.0, 0.0, 1.0), 8, (0.0, 1.0)),
        spatial=spatial,
        edge_energy=float(np.mean(gradient)),
        luma_std=float(np.std(luma)),
    )


def _hist_intersection(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    return float(np.minimum(left, right).sum())


def _spatial_similarity(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    if left.shape != right.shape:
        return 0.0
    left_flat = left.ravel()
    right_flat = right.ravel()
    left_std = float(left_flat.std())
    right_std = float(right_flat.std())
    if left_std <= 1e-8 or right_std <= 1e-8:
        return 0.5
    correlation = float(np.dot(left_flat, right_flat) / (left_flat.size * left_std * right_std))
    return float(np.clip((correlation + 1.0) * 0.5, 0.0, 1.0))


def _feature_similarity(left: _FastFeature, right: _FastFeature) -> float:
    luminance = _hist_intersection(left.luma_hist, right.luma_hist)
    chroma = _hist_intersection(left.chroma_hist, right.chroma_hist)
    gradient = _hist_intersection(left.gradient_hist, right.gradient_hist)
    spatial = _spatial_similarity(left.spatial, right.spatial)
    edge_ratio = (left.edge_energy + 1e-6) / (right.edge_energy + 1e-6)
    edge_similarity = float(np.exp(-abs(np.log(edge_ratio))))
    return float(
        np.clip(
            0.32 * luminance
            + 0.22 * chroma
            + 0.20 * gradient
            + 0.16 * spatial
            + 0.10 * edge_similarity,
            0.0,
            1.0,
        )
    )


def _build_features(source: SourceInfo, limit_seconds: float) -> list[_FastFeature]:
    fps = float(source.selected_stream.fps if source.selected_stream else 0.0)
    if fps <= 0:
        raise SynchronizationError("Fast sync source has invalid FPS")
    with tempfile.TemporaryDirectory(prefix="openmatte-fast-sync-") as directory:
        clip_path = Path(directory) / "keyframes.mkv"
        _extract_keyframe_clip(source, limit_seconds, clip_path)
        timestamps = _clip_timestamps(clip_path, fps)
        frames = _decode_clip_rgb(clip_path, source)
    count = min(len(timestamps), len(frames))
    features = [
        _feature_from_frame(frames[index], timestamps[index], round(timestamps[index] * fps))
        for index in range(count)
        if 0.0 <= timestamps[index] <= limit_seconds
    ]
    if len(features) < 3:
        raise SynchronizationError(
            f"Fast sync found too few keyframe thumbnails for {source.path}"
        )
    for index, feature in enumerate(features):
        novelty = 0.0
        if index > 0:
            novelty = 1.0 - _feature_similarity(features[index - 1], feature)
        feature.characteristic_score = float(
            0.45 * min(1.0, feature.luma_std * 4.0)
            + 0.30 * min(1.0, feature.edge_energy * 12.0)
            + 0.25 * np.clip(novelty, 0.0, 1.0)
        )
    return features


def _select_representative(
    features: list[_FastFeature],
    start_seconds: float,
    end_seconds: float,
) -> list[_FastFeature]:
    selected: list[_FastFeature] = []
    start = float(start_seconds)
    end = max(start, float(end_seconds))
    edges = np.linspace(start, end, _FAST_POINT_COUNT + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        candidates = [item for item in features if lower <= item.timestamp <= upper]
        if not candidates:
            candidates = [
                min(
                    features,
                    key=lambda item: abs(item.timestamp - (lower + upper) * 0.5),
                )
            ]
        selected.append(max(candidates, key=lambda item: item.characteristic_score))
    unique: dict[int, _FastFeature] = {item.frame_index: item for item in selected}
    return list(sorted(unique.values(), key=lambda item: item.timestamp))


def _cluster_votes(
    votes: list[tuple[int, float]],
    tolerance_frames: int,
) -> list[tuple[int, float]]:
    clusters: list[list[tuple[int, float]]] = []
    for offset, weight in sorted(votes, key=lambda item: -item[1]):
        placed = False
        for cluster in clusters:
            weighted_center = sum(
                value * score for value, score in cluster
            ) / sum(score for _, score in cluster)
            if abs(offset - weighted_center) <= tolerance_frames:
                cluster.append((offset, weight))
                placed = True
                break
        if not placed:
            clusters.append([(offset, weight)])
    ranked: list[tuple[int, float]] = []
    for cluster in clusters:
        total = sum(weight for _, weight in cluster)
        center = int(round(sum(offset * weight for offset, weight in cluster) / total))
        ranked.append((center, total))
    return sorted(ranked, key=lambda item: -item[1])


def _sparse_candidates(
    hdr_features: list[_FastFeature],
    om_features: list[_FastFeature],
    hdr_fps: float,
    max_offset_frames: int,
) -> tuple[list[tuple[int, float]], float]:
    votes: list[tuple[int, float]] = []
    sparse_scores: dict[int, list[float]] = {}
    for hdr in hdr_features:
        matches: list[tuple[float, _FastFeature]] = []
        for om in om_features:
            offset = om.frame_index - hdr.frame_index
            if abs(offset) <= max_offset_frames:
                matches.append((_feature_similarity(hdr, om), om))
        matches.sort(key=lambda item: -item[0])
        for rank, (similarity, om) in enumerate(matches[:_FAST_TOP_MATCHES]):
            if similarity < _FAST_MIN_SIMILARITY:
                continue
            offset = om.frame_index - hdr.frame_index
            weight = similarity * (1.0 - 0.08 * rank) * (0.5 + hdr.characteristic_score)
            votes.append((offset, weight))
            sparse_scores.setdefault(offset, []).append(similarity)
    if not votes:
        return [], 0.0
    tolerance = max(2, int(round(hdr_fps * 0.25)))
    clusters = _cluster_votes(votes, tolerance)
    total_weight = sum(weight for _, weight in clusters)
    consensus = clusters[0][1] / total_weight if total_weight > 0 else 0.0
    return clusters[:5], float(np.clip(consensus, 0.0, 1.0))


def _sparse_score_for_offset(
    hdr_features: list[_FastFeature],
    om_features: list[_FastFeature],
    offset: int,
    hdr_fps: float,
) -> float:
    """Return the existing sparse similarity score for one candidate."""
    tolerance = max(2, round(hdr_fps * 0.25))
    return float(
        np.mean(
            [
                max(
                    (
                        _feature_similarity(hdr, om)
                        for om in om_features
                        if abs(om.frame_index - hdr.frame_index - offset)
                        <= tolerance
                    ),
                    default=0.0,
                )
                for hdr in hdr_features
            ]
        )
    )


def _best_sparse_match(
    hdr_feature: _FastFeature,
    om_features: list[_FastFeature],
    candidates: list[tuple[int, float]],
    hdr_fps: float,
) -> tuple[int | None, float]:
    """Find a sample's best offset among the already generated candidates."""
    tolerance = max(2, round(hdr_fps * 0.25))
    matches: list[tuple[int, float]] = []
    for offset, _weight in candidates:
        score = max(
            (
                _feature_similarity(hdr_feature, om)
                for om in om_features
                if abs(om.frame_index - hdr_feature.frame_index - offset)
                <= tolerance
            ),
            default=0.0,
        )
        matches.append((int(offset), float(score)))
    if not matches:
        return None, 0.0
    return max(matches, key=lambda item: item[1])


def _candidate_diagnostics(
    candidates: list[tuple[int, float]],
) -> tuple[list[dict[str, float | int]], float, float, float]:
    """Expose comparable top-five vote scores without changing candidate ranking."""
    top_weight = float(candidates[0][1]) if candidates else 0.0
    total_top_weight = sum(float(weight) for _offset, weight in candidates)
    best_score = (
        float(candidates[0][1]) / total_top_weight
        if candidates and total_top_weight > 0
        else 0.0
    )
    second_score = (
        float(candidates[1][1]) / total_top_weight
        if len(candidates) > 1 and total_top_weight > 0
        else 0.0
    )
    diagnostics: list[dict[str, float | int]] = []
    for rank, (offset, weight) in enumerate(candidates, start=1):
        diagnostics.append(
            {
                "rank": rank,
                "offset": int(offset),
                "score": float(weight / total_top_weight)
                if total_top_weight > 0
                else 0.0,
                "relative_score": float(weight / top_weight)
                if top_weight > 0
                else 0.0,
                "vote_weight": float(weight),
            }
        )
    return diagnostics, best_score, second_score, best_score - second_score


def _gray_frame_similarity(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> float:
    min_height = min(left.shape[0], right.shape[0])
    min_width = min(left.shape[1], right.shape[1])
    left_small = cv2.resize(
        left[:min_height, :min_width].astype(np.float32),
        (32, 18),
        interpolation=cv2.INTER_AREA,
    )
    right_small = cv2.resize(
        right[:min_height, :min_width].astype(np.float32),
        (32, 18),
        interpolation=cv2.INTER_AREA,
    )
    lum_left = _normalise_histogram(left_small, 16, (0.0, 1.0))
    lum_right = _normalise_histogram(right_small, 16, (0.0, 1.0))
    gx_left = np.diff(left_small, axis=1, prepend=left_small[:, :1])
    gy_left = np.diff(left_small, axis=0, prepend=left_small[:1, :])
    gx_right = np.diff(right_small, axis=1, prepend=right_small[:, :1])
    gy_right = np.diff(right_small, axis=0, prepend=right_small[:1, :])
    grad_left = _normalise_histogram(np.hypot(gx_left, gy_left), 8, (0.0, 1.0))
    grad_right = _normalise_histogram(np.hypot(gx_right, gy_right), 8, (0.0, 1.0))
    energy_left = float(np.mean(np.hypot(gx_left, gy_left)))
    energy_right = float(np.mean(np.hypot(gx_right, gy_right)))
    ratio = (energy_left + 1e-6) / (energy_right + 1e-6)
    energy_similarity = float(np.exp(-abs(np.log(ratio))))
    return float(
        np.clip(
            0.55 * _hist_intersection(lum_left, lum_right)
            + 0.30 * _hist_intersection(grad_left, grad_right)
            + 0.15 * energy_similarity,
            0.0,
            1.0,
        )
    )


def _verify_candidate(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    anchors: list[_FastFeature],
    candidate_offset: int,
    hdr_fps: float,
    om_fps: float,
) -> tuple[float, float, list[dict[str, float | int | bool | None]]]:
    """Compare several consecutive-frame windows for one offset candidate."""
    hdr_ordinal = _video_ordinal(hdr_source)
    om_ordinal = _video_ordinal(om_source)
    half = _FAST_WINDOW_FRAMES // 2
    anchor_scores: list[float] = []
    anchor_records: list[dict[str, float | int | bool | None]] = []
    for anchor in anchors[:_FAST_VERIFY_ANCHORS]:
        hdr_start = max(0, anchor.frame_index - half)
        om_start = hdr_start + candidate_offset
        if om_start < 0:
            anchor_records.append(
                {
                    "hdr_frame": int(anchor.frame_index),
                    "timestamp": float(anchor.timestamp),
                    "candidate_offset": int(candidate_offset),
                    "score": None,
                    "valid_frame_samples": 0,
                    "independent": False,
                }
            )
            continue
        hdr_frames = extract_segment_at_time_grayscale(
            hdr_source.path,
            hdr_start / hdr_fps,
            _FAST_WINDOW_FRAMES,
            stream_index=hdr_ordinal,
            width=_FAST_VERIFY_WIDTH,
            timeout=90,
        )
        om_frames = extract_segment_at_time_grayscale(
            om_source.path,
            om_start / om_fps,
            _FAST_WINDOW_FRAMES,
            stream_index=om_ordinal,
            width=_FAST_VERIFY_WIDTH,
            timeout=90,
        )
        frame_scores = [
            _gray_frame_similarity(left, right)
            for left, right in zip(hdr_frames, om_frames)
            if left is not None and right is not None
        ]
        mean_score = float(np.mean(frame_scores)) if frame_scores else None
        anchor_records.append(
            {
                "hdr_frame": int(anchor.frame_index),
                "timestamp": float(anchor.timestamp),
                "candidate_offset": int(candidate_offset),
                "score": mean_score,
                "valid_frame_samples": len(frame_scores),
                "independent": bool(frame_scores),
            }
        )
        if mean_score is not None:
            anchor_scores.append(mean_score)
    if not anchor_scores:
        return 0.0, 0.0, anchor_records
    agreement = sum(score >= 0.55 for score in anchor_scores) / len(anchor_scores)
    return float(np.mean(anchor_scores)), float(agreement), anchor_records


def find_fast_global_offset(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    config: SyncConfig | None = None,
    *,
    progress_callback: _PROGRESS | None = None,
) -> SyncModel:
    """Find a fast first-ten-minute sync proposal without scanning the film."""
    if config is None:
        config = SyncConfig()
    hdr_stream = hdr_source.selected_stream
    om_stream = om_source.selected_stream
    hdr_fps = float(hdr_stream.fps if hdr_stream else 0.0)
    om_fps = float(om_stream.fps if om_stream else 0.0)
    if hdr_fps <= 0 or om_fps <= 0:
        raise SynchronizationError(
            f"Invalid FPS for fast synchronization: HDR={hdr_fps}, OM={om_fps}."
        )
    hdr_duration = _source_duration(hdr_source)
    om_duration = _source_duration(om_source)
    analysis_limit = min(FAST_SYNC_MAX_SECONDS, hdr_duration, om_duration)
    if analysis_limit < 30.0:
        raise SynchronizationError(
            f"Fast sync requires at least 30 seconds in both sources; range={analysis_limit:.2f}s"
        )
    max_offset_frames = int(config.search_range_seconds * hdr_fps)
    safe_margin = min(config.search_range_seconds, analysis_limit * 0.2)
    sample_limit = analysis_limit - safe_margin
    if sample_limit <= safe_margin:
        safe_margin = max(2.0, analysis_limit * 0.05)
        sample_limit = analysis_limit - safe_margin

    _progress(progress_callback, "Fast Auto Sync: extracting first 10 min thumbnails")
    hdr_features = _build_features(hdr_source, analysis_limit)
    _progress(progress_callback, "Fast Auto Sync: extracting OM thumbnails")
    om_features = _build_features(om_source, analysis_limit)
    hdr_features = [item for item in hdr_features if safe_margin <= item.timestamp <= sample_limit]
    representatives = _select_representative(
        hdr_features,
        safe_margin,
        sample_limit,
    )
    if len(representatives) < 3:
        raise SynchronizationError("Fast sync found too few representative HDR points")

    _progress(progress_callback, "Fast Auto Sync: voting offset candidates")
    candidates, consensus = _sparse_candidates(
        representatives,
        om_features,
        hdr_fps,
        max_offset_frames,
    )
    if not candidates:
        raise SynchronizationError("Fast sync found no valid offset candidates")

    _progress(progress_callback, "Fast Auto Sync: verifying consecutive frames")
    candidate_entries, best_candidate_score, second_candidate_score, score_margin = (
        _candidate_diagnostics(candidates)
    )
    verification_by_offset: dict[
        int, tuple[float, float, list[dict[str, float | int | bool | None]]]
    ] = {}
    ranked_verification: list[tuple[int, float, float, float]] = []
    verification_anchors = sorted(
        representatives,
        key=lambda item: -item.characteristic_score,
    )
    for offset, _weight in candidates[:3]:
        sparse_score = _sparse_score_for_offset(
            representatives,
            om_features,
            offset,
            hdr_fps,
        )
        detailed, agreement, anchor_records = _verify_candidate(
            hdr_source,
            om_source,
            verification_anchors,
            offset,
            hdr_fps,
            om_fps,
        )
        verification_by_offset[int(offset)] = (
            detailed,
            agreement,
            anchor_records,
        )
        ranked_verification.append((offset, sparse_score, detailed, agreement))
    best_offset, sparse_score, detailed_score, agreement = max(
        ranked_verification,
        key=lambda item: 0.25 * item[1] + 0.55 * item[2] + 0.20 * item[3],
    )
    confidence = float(
        np.clip(
            0.25 * consensus
            + 0.25 * sparse_score
            + 0.35 * detailed_score
            + 0.15 * agreement,
            0.0,
            1.0,
        )
    )
    status = SyncStatus.LOCKED if confidence >= config.min_confidence else SyncStatus.FAILED
    frame_locked = status == SyncStatus.LOCKED

    _progress(progress_callback, "Fast Auto Sync: post-verifying selected offset")
    post_detailed_score, post_similarity_agreement, post_anchor_records = (
        _verify_candidate(
            hdr_source,
            om_source,
            verification_anchors,
            best_offset,
            hdr_fps,
            om_fps,
        )
    )

    selection_scores = {
        int(offset): 0.25 * sparse_value
        + 0.55 * detailed_value
        + 0.20 * similarity_agreement
        for offset, sparse_value, detailed_value, similarity_agreement in ranked_verification
    }
    for entry in candidate_entries:
        candidate_offset = int(entry["offset"])
        verification = verification_by_offset.get(candidate_offset)
        entry["verified"] = verification is not None
        entry["selection_score"] = (
            float(selection_scores[candidate_offset])
            if candidate_offset in selection_scores
            else None
        )
        entry["detailed_score"] = (
            float(verification[0]) if verification is not None else None
        )
        entry["similarity_agreement"] = (
            float(verification[1]) if verification is not None else None
        )

    sample_diagnostics: list[dict[str, float | int | bool | None]] = []
    for sample_index, representative in enumerate(representatives, start=1):
        sample_offset, sample_score = _best_sparse_match(
            representative,
            om_features,
            candidates,
            hdr_fps,
        )
        independent = (
            sample_offset is not None
            and sample_score >= _FAST_MIN_SIMILARITY
        )
        sample_diagnostics.append(
            {
                "sample": sample_index,
                "hdr_frame": int(representative.frame_index),
                "timestamp": float(representative.timestamp),
                "best_offset": sample_offset,
                "best_score": float(sample_score),
                "independent": bool(independent),
            }
        )

    independent_offsets = [
        int(item["best_offset"])
        for item in sample_diagnostics
        if item["independent"] and item["best_offset"] is not None
    ]
    independent_sample_count = len(independent_offsets)
    agreement_count = sum(
        abs(offset - best_offset) <= _FAST_DIAGNOSTIC_MAX_AGREEMENT_TOLERANCE_FRAMES
        for offset in independent_offsets
    )
    agreement_percentage = (
        agreement_count / independent_sample_count
        if independent_sample_count
        else 0.0
    )
    offset_spread_frames = (
        float(max(independent_offsets) - min(independent_offsets))
        if independent_offsets
        else None
    )
    spread_quality = (
        1.0
        if offset_spread_frames is not None
        and offset_spread_frames <= _FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES
        else 0.0
    )
    margin_quality = float(
        np.clip(score_margin / _FAST_DIAGNOSTIC_MIN_MARGIN, 0.0, 1.0)
    )
    verification_independent_count = sum(
        bool(item["independent"]) for item in post_anchor_records
    )
    verification_valid_frame_samples = sum(
        int(item["valid_frame_samples"]) for item in post_anchor_records
    )
    verification_coverage = float(
        np.clip(
            verification_independent_count / _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES,
            0.0,
            1.0,
        )
    )
    fast_confidence = float(
        np.clip(
            0.10 * consensus
            + 0.10 * best_candidate_score
            + 0.15 * margin_quality
            + 0.25 * agreement_percentage
            + 0.10 * spread_quality
            + 0.20 * post_detailed_score
            + 0.10 * verification_coverage,
            0.0,
            1.0,
        )
    )

    candidate_best_offset = int(candidates[0][0])
    candidate_rank = next(
        (
            int(entry["rank"])
            for entry in candidate_entries
            if int(entry["offset"]) == int(best_offset)
        ),
        None,
    )
    fast_diagnostic_status = (
        "FAST_LOCKED"
        if (
            fast_confidence >= _FAST_DIAGNOSTIC_MIN_CONFIDENCE
            and independent_sample_count
            >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES
            and agreement_count >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES
            and agreement_percentage == 1.0
            and offset_spread_frames is not None
            and offset_spread_frames <= _FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES
            and score_margin >= _FAST_DIAGNOSTIC_MIN_MARGIN
            and verification_independent_count
            >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES
            and verification_valid_frame_samples
            >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES * _FAST_WINDOW_FRAMES
        )
        else "FAST_REVIEW"
    )
    fast_diagnostics = {
        "schema_version": 1,
        "analysis_limit_seconds": float(analysis_limit),
        "candidate_score_basis": "top5_cluster_vote_share",
        "best_offset": candidate_best_offset,
        "best_score": float(best_candidate_score),
        "second_best_offset": (
            int(candidates[1][0]) if len(candidates) > 1 else None
        ),
        "second_score": float(second_candidate_score),
        "score_margin": float(score_margin),
        "selected_offset": int(best_offset),
        "selected_candidate_rank": candidate_rank,
        "candidates": candidate_entries,
        "anchors": sample_diagnostics,
        "agreement_count": int(agreement_count),
        "agreement_percentage": float(agreement_percentage),
        "offset_spread_frames": offset_spread_frames,
        "independent_sample_count": int(independent_sample_count),
        "minimum_independent_samples": _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES,
        "verification_anchors": post_anchor_records,
        "verification": {
            "consecutive": True,
            "window_frames": _FAST_WINDOW_FRAMES,
            "proxy_width": _FAST_VERIFY_WIDTH,
            "anchor_limit": _FAST_VERIFY_ANCHORS,
            "similarity_threshold": 0.55,
            "verified_candidate_count": len(ranked_verification),
            "post_selection_score": float(post_detailed_score),
            "post_selection_similarity_agreement": float(
                post_similarity_agreement
            ),
            "post_selection_independent_anchors": int(
                verification_independent_count
            ),
            "post_selection_valid_frame_samples": int(
                verification_valid_frame_samples
            ),
        },
        "fast_confidence_components": {
            "candidate_consensus": float(consensus),
            "best_candidate_score": float(best_candidate_score),
            "margin_quality": margin_quality,
            "sample_agreement": float(agreement_percentage),
            "spread_quality": spread_quality,
            "post_selection_verification": float(post_detailed_score),
            "verification_coverage": verification_coverage,
        },
        "fast_lock_policy": {
            "status": "LOCKED",
            "legacy_confidence_unchanged": True,
            "min_fast_confidence": _FAST_DIAGNOSTIC_MIN_CONFIDENCE,
            "min_independent_samples": _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES,
            "min_agreement_percentage": 1.0,
            "max_offset_spread_frames": _FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES,
            "min_score_margin": _FAST_DIAGNOSTIC_MIN_MARGIN,
        },
    }

    if not frame_locked:
        logger.warning(
            "Fast sync confidence %.4f is below %.4f; result is not frame-locked",
            confidence,
            config.min_confidence,
        )
    offset_seconds = best_offset / hdr_fps
    _progress(
        progress_callback,
        f"Fast Auto Sync: {status.value}, offset={best_offset}, confidence={confidence:.3f}",
    )
    return SyncModel(
        frame_offset=int(best_offset),
        confidence=confidence,
        status=status,
        frame_locked=frame_locked,
        offset_seconds=offset_seconds,
        method="fast_first_10m_keyframe_features",
        fast_confidence=fast_confidence,
        fast_diagnostic_status=fast_diagnostic_status,
        fast_diagnostics=fast_diagnostics,
    )
