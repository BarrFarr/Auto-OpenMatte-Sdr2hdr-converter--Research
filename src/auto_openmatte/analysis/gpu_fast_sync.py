"""Bounded GPU-only Fast Auto Sync offset proposal for the first ten minutes.

The reference synchronizer lives in :mod:`auto_openmatte.analysis.sync` and is
intentionally not called here.  This strategy uses V5 native NVDEC sessions,
keeps all image data in CUDA device memory, and transfers only scalar
reductions plus small top-K tables to the host.

Two properties of the native bridge shape this implementation.

*Cost*: HDR frames are resampled by the bridge to a narrow working plane, so a
decoded HDR frame costs about 1.6 ms, while an OM frame arrives at full source
geometry and costs about 5.9 ms.  Opening a decoder with a seek costs 50-600 ms.
The search therefore scans contiguously on the cheap HDR side and samples the
expensive OM side at a few anchors only.

*Domain*: the bridge emits linear light for both sources, but on different
scales - HDR is normalised against the PQ peak while OM is normalised against
the BT.1886 diffuse white.  Measured on real material, HDR luma occupied
0.00002-0.009 while the matching OM luma occupied 0-0.62, so fixed-range
histograms of raw luma cannot be compared at all.  Descriptors are therefore
built from log luminance standardised per frame, which is invariant to both a
scale factor and a gamma difference, and the OM frame is centre-cropped to the
HDR field of view before pooling so both grids describe the same framing.

The native runtime is an external deployment dependency.  If it is not
available, this module raises an explicit synchronization error; it never
silently falls back to the CPU fast synchronizer.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from auto_openmatte.core.config import SyncConfig
from auto_openmatte.core.exceptions import SynchronizationError
from auto_openmatte.core.models import SourceInfo, SyncModel, SyncStatus
from auto_openmatte.utils.ffmpeg import get_media_tool_config

GPU_FAST_MAX_SECONDS = 10 * 60.0

# Descriptor geometry. Both sources are reduced to the same working plane so
# their descriptors describe the same framing at the same resolution.
_DESCRIPTOR_WIDTH = 256
_GRID_WIDTH = 16
_GRID_HEIGHT = 9
_LUMA_BINS = 16
_CHROMA_BINS = 8
_GRADIENT_BINS = 8
_POOLED_VALUES = _GRID_WIDTH * _GRID_HEIGHT
_DESCRIPTOR_LENGTH = (
    _LUMA_BINS + 2 * _CHROMA_BINS + _GRADIENT_BINS + _POOLED_VALUES + 3
)
_POOLED_OFFSET = _LUMA_BINS + 2 * _CHROMA_BINS + _GRADIENT_BINS
_SCALAR_OFFSET = _POOLED_OFFSET + _POOLED_VALUES

# Search shape.
_ANCHOR_COUNT = 5
_ANCHOR_WINDOW_FRAMES = 9
_REFINE_HALF_FRAMES = 4
_CONSENSUS_CANDIDATES = 3
# The coarse descriptor localises the matching shot but saturates inside it, so
# a second pass correlates the full working planes to reach frame resolution.
_FINE_HALF_FRAMES = 16
_FINE_MAX_FRAMES = 257
_FINE_PEAK_GUARD_FRAMES = 2
# A correlation peak that barely exceeds its neighbours comes from a static
# shot, where no method can resolve a single frame; such anchors do not vote.
_FINE_MIN_PEAK_SHARPNESS = 0.005
_FINE_NEIGHBOURHOOD_FRAMES = 4
_GPU_TOP_K = 5
_CANDIDATE_POOL = 256
_CANDIDATE_MIN_SEPARATION_FRAMES = 12
_GPU_MIN_SIMILARITY = 0.35
_PROGRESS_FRAME_INTERVAL = 600

# Diagnostic gates. These keep the existing Fast Auto Sync contract unchanged.
_GPU_VERIFY_ANCHORS = 3
_GPU_VERIFY_WINDOW_FRAMES = 9
_GPU_VERIFY_SIMILARITY = 0.55
_FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES = 3
_FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES = 1
_FAST_DIAGNOSTIC_MAX_AGREEMENT_TOLERANCE_FRAMES = 1
_FAST_DIAGNOSTIC_MIN_MARGIN = 0.10
_FAST_DIAGNOSTIC_MIN_CONFIDENCE = 0.80

_PROGRESS = Callable[[str], None]

_KERNEL_SOURCE = r"""
#define LUMA_BINS 16
#define CHROMA_BINS 8
#define GRADIENT_BINS 8
#define GRID_WIDTH 16
#define GRID_HEIGHT 9
#define POOLED_VALUES (GRID_WIDTH * GRID_HEIGHT)
#define CHROMA_OFFSET LUMA_BINS
#define GRADIENT_OFFSET (LUMA_BINS + 2 * CHROMA_BINS)
#define POOLED_OFFSET (GRADIENT_OFFSET + GRADIENT_BINS)
#define SCALAR_OFFSET (POOLED_OFFSET + POOLED_VALUES)
#define DESCRIPTOR_LENGTH (SCALAR_OFFSET + 3)
#define LOG_EPSILON 1.0e-6f
#define STANDARD_RANGE 3.0f

__device__ __forceinline__ int unit_bin(float value, int bins) {
    int bin = (int)(value * (float)bins);
    if (bin < 0) {
        bin = 0;
    }
    if (bin >= bins) {
        bin = bins - 1;
    }
    return bin;
}

__device__ __forceinline__ float standard_position(float value, float mean, float deviation) {
    return ((value - mean) / deviation + STANDARD_RANGE) / (2.0f * STANDARD_RANGE);
}

/* Crop to the reference field of view, box-average to the working plane and
   convert to log luminance plus log chroma ratios. Working in log space turns a
   scale difference into an offset and a gamma difference into a factor, both of
   which the per-frame standardisation below removes. The full 4K OM payload is
   read exactly once and never leaves device memory. */
extern "C" __global__ void openmatte_prepare_planes(
    const float *__restrict__ source,
    int source_width,
    int source_height,
    int crop_left,
    int crop_top,
    int factor_x,
    int factor_y,
    int width,
    int height,
    float *__restrict__ log_luma,
    float *__restrict__ chroma_rg,
    float *__restrict__ chroma_gb
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (index >= total) {
        return;
    }
    int first_x = crop_left + (index % width) * factor_x;
    int first_y = crop_top + (index / width) * factor_y;
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
    int samples = 0;
    for (int offset_y = 0; offset_y < factor_y; ++offset_y) {
        int source_y = first_y + offset_y;
        if (source_y < 0 || source_y >= source_height) {
            continue;
        }
        const float *row = source + ((size_t)source_y * (size_t)source_width) * 3;
        for (int offset_x = 0; offset_x < factor_x; ++offset_x) {
            int source_x = first_x + offset_x;
            if (source_x < 0 || source_x >= source_width) {
                continue;
            }
            const float *pixel = row + (size_t)source_x * 3;
            red += pixel[0];
            green += pixel[1];
            blue += pixel[2];
            ++samples;
        }
    }
    float scale = samples > 0 ? 1.0f / (float)samples : 0.0f;
    red = fmaxf(red * scale, 0.0f) + LOG_EPSILON;
    green = fmaxf(green * scale, 0.0f) + LOG_EPSILON;
    blue = fmaxf(blue * scale, 0.0f) + LOG_EPSILON;
    float luma = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
    float log_red = logf(red);
    float log_green = logf(green);
    float log_blue = logf(blue);
    log_luma[index] = logf(luma);
    chroma_rg[index] = log_red - log_green;
    chroma_gb[index] = log_green - log_blue;
}

/* First and second raw moments of the three working planes. */
extern "C" __global__ void openmatte_plane_moments(
    const float *__restrict__ log_luma,
    const float *__restrict__ chroma_rg,
    const float *__restrict__ chroma_gb,
    int total,
    float *__restrict__ moments
) {
    __shared__ float shared_moments[6];
    if (threadIdx.x < 6) {
        shared_moments[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    float luma_sum = 0.0f;
    float luma_square = 0.0f;
    float rg_sum = 0.0f;
    float rg_square = 0.0f;
    float gb_sum = 0.0f;
    float gb_square = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += blockDim.x * gridDim.x) {
        float luma = log_luma[index];
        float rg = chroma_rg[index];
        float gb = chroma_gb[index];
        luma_sum += luma;
        luma_square += luma * luma;
        rg_sum += rg;
        rg_square += rg * rg;
        gb_sum += gb;
        gb_square += gb * gb;
    }
    atomicAdd(&shared_moments[0], luma_sum);
    atomicAdd(&shared_moments[1], luma_square);
    atomicAdd(&shared_moments[2], rg_sum);
    atomicAdd(&shared_moments[3], rg_square);
    atomicAdd(&shared_moments[4], gb_sum);
    atomicAdd(&shared_moments[5], gb_square);
    __syncthreads();
    if (threadIdx.x < 6) {
        atomicAdd(&moments[threadIdx.x], shared_moments[threadIdx.x]);
    }
}

/* Accumulate one complete frame descriptor from the standardised planes:
   luminance histogram, two chroma histograms, gradient-energy histogram,
   pooled luma grid, and the scalars used for edge energy and contrast. */
extern "C" __global__ void openmatte_descriptor(
    const float *__restrict__ log_luma,
    const float *__restrict__ chroma_rg,
    const float *__restrict__ chroma_gb,
    int width,
    int height,
    int pool_width,
    int pool_height,
    const float *__restrict__ moments,
    float *__restrict__ descriptor
) {
    int total = width * height;
    float inverse = 1.0f / (float)total;
    float luma_mean = moments[0] * inverse;
    float luma_deviation = sqrtf(
        fmaxf(moments[1] * inverse - luma_mean * luma_mean, 1.0e-12f)
    );
    float rg_mean = moments[2] * inverse;
    float rg_deviation = sqrtf(fmaxf(moments[3] * inverse - rg_mean * rg_mean, 1.0e-12f));
    float gb_mean = moments[4] * inverse;
    float gb_deviation = sqrtf(fmaxf(moments[5] * inverse - gb_mean * gb_mean, 1.0e-12f));

    __shared__ float shared_values[DESCRIPTOR_LENGTH];
    for (int index = threadIdx.x; index < DESCRIPTOR_LENGTH; index += blockDim.x) {
        shared_values[index] = 0.0f;
    }
    __syncthreads();

    int grid_pixels_x = GRID_WIDTH * pool_width;
    int grid_pixels_y = GRID_HEIGHT * pool_height;
    float local_gradient = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += blockDim.x * gridDim.x) {
        int x = index % width;
        int y = index / width;
        float value = log_luma[index];
        float standardised = (value - luma_mean) / luma_deviation;
        float gradient_x = x > 0 ? (value - log_luma[index - 1]) / luma_deviation : 0.0f;
        float gradient_y = y > 0 ? (value - log_luma[index - width]) / luma_deviation : 0.0f;
        float gradient = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

        int luma_slot = unit_bin(
            standard_position(value, luma_mean, luma_deviation), LUMA_BINS
        );
        atomicAdd(&shared_values[luma_slot], 1.0f);
        atomicAdd(
            &shared_values[
                CHROMA_OFFSET
                + unit_bin(standard_position(chroma_rg[index], rg_mean, rg_deviation), CHROMA_BINS)
            ],
            1.0f
        );
        atomicAdd(
            &shared_values[
                CHROMA_OFFSET + CHROMA_BINS
                + unit_bin(standard_position(chroma_gb[index], gb_mean, gb_deviation), CHROMA_BINS)
            ],
            1.0f
        );
        atomicAdd(
            &shared_values[GRADIENT_OFFSET + unit_bin(gradient * 0.25f, GRADIENT_BINS)],
            1.0f
        );
        if (x < grid_pixels_x && y < grid_pixels_y) {
            int cell = (y / pool_height) * GRID_WIDTH + (x / pool_width);
            atomicAdd(&shared_values[POOLED_OFFSET + cell], standardised);
        }
        local_gradient += gradient;
    }
    atomicAdd(&shared_values[SCALAR_OFFSET], local_gradient);
    __syncthreads();

    for (int index = threadIdx.x; index <= SCALAR_OFFSET; index += blockDim.x) {
        float value = shared_values[index];
        if (value != 0.0f) {
            atomicAdd(&descriptor[index], value);
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        descriptor[SCALAR_OFFSET + 1] = luma_deviation;
        descriptor[SCALAR_OFFSET + 2] = luma_mean;
    }
}
"""


class GpuFastSyncUnavailable(SynchronizationError):
    """The explicitly requested GPU fast-sync runtime is unavailable."""


@dataclass(frozen=True)
class _GpuRuntime:
    cupy: Any
    native: Any
    ffmpeg_bin: Path
    bridge_path: Path
    tools_dir: Path
    device_index: int
    prepare_kernel: Any
    moments_kernel: Any
    descriptor_kernel: Any


@dataclass(frozen=True)
class _Geometry:
    """Working-plane geometry used for descriptor extraction."""

    width: int
    height: int
    crop_left: int
    crop_top: int
    factor_x: int
    factor_y: int
    pool_width: int
    pool_height: int

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def pool_area(self) -> int:
        return self.pool_width * self.pool_height


@dataclass
class _Bank:
    """A device matrix of raw descriptor accumulators plus reusable scratch."""

    cupy: Any
    geometry: _Geometry
    matrix: Any
    log_luma: Any
    chroma_rg: Any
    chroma_gb: Any
    moments: Any
    planes: Any = None
    rows: dict[int, int] = field(default_factory=dict)
    count: int = 0

    @classmethod
    def create(
        cls,
        cupy: Any,
        geometry: _Geometry,
        capacity: int,
        *,
        store_planes: bool = False,
    ) -> "_Bank":
        pixels = geometry.pixel_count
        return cls(
            cupy=cupy,
            geometry=geometry,
            matrix=cupy.zeros(
                (max(1, int(capacity)), _DESCRIPTOR_LENGTH), dtype=cupy.float32
            ),
            log_luma=cupy.empty((pixels,), dtype=cupy.float32),
            chroma_rg=cupy.empty((pixels,), dtype=cupy.float32),
            chroma_gb=cupy.empty((pixels,), dtype=cupy.float32),
            moments=cupy.zeros((6,), dtype=cupy.float32),
            planes=(
                cupy.empty((max(1, int(capacity)), pixels), dtype=cupy.float32)
                if store_planes
                else None
            ),
        )

    def reserve(self, frame_index: int) -> int | None:
        """Return the row for a frame, or ``None`` when it is already present."""
        if frame_index in self.rows:
            return None
        if self.count >= int(self.matrix.shape[0]):
            raise GpuFastSyncUnavailable(
                "GPU Fast Auto Sync descriptor bank capacity was exceeded"
            )
        row = self.count
        self.rows[int(frame_index)] = row
        self.count += 1
        return row

    def row_of(self, frame_index: int) -> int | None:
        return self.rows.get(int(frame_index))


@dataclass(frozen=True)
class _Descriptors:
    """Normalized descriptor components; every array stays on the device."""

    cupy: Any
    luma: Any
    chroma: Any
    gradient: Any
    spatial: Any
    edge: Any
    contrast: Any

    @classmethod
    def from_raw(cls, cupy: Any, raw: Any, geometry: _Geometry) -> "_Descriptors":
        pixels = float(geometry.pixel_count)
        pooled = raw[:, _POOLED_OFFSET:_SCALAR_OFFSET] / float(geometry.pool_area)
        centered = pooled - pooled.mean(axis=1, keepdims=True)
        deviation = cupy.sqrt((centered * centered).mean(axis=1, keepdims=True))
        return cls(
            cupy=cupy,
            luma=raw[:, 0:_LUMA_BINS] / pixels,
            chroma=raw[:, _LUMA_BINS:_LUMA_BINS + 2 * _CHROMA_BINS] / pixels,
            gradient=raw[
                :, _LUMA_BINS + 2 * _CHROMA_BINS:_POOLED_OFFSET
            ] / pixels,
            spatial=centered / cupy.maximum(deviation, cupy.float32(1e-6)),
            edge=raw[:, _SCALAR_OFFSET] / pixels,
            contrast=raw[:, _SCALAR_OFFSET + 1],
        )

    def characteristic_scores(self) -> Any:
        """Rank how distinctive frames are; used only to order anchors."""
        cupy = self.cupy
        return cupy.clip(
            0.45 * cupy.minimum(self.contrast / 3.0, 1.0)
            + 0.30 * cupy.minimum(self.edge * 4.0, 1.0),
            0.0,
            1.0,
        )


@dataclass
class _Anchor:
    """One independent OM sampling point and its resolved vote."""

    sample: int
    center_frame: int
    timestamp: float
    characteristic: float = 0.0
    resolvability: float = 0.0
    best_offset: int | None = None
    best_score: float = 0.0
    source_candidate: int | None = None


@dataclass
class _Cluster:
    offset: int
    weight: float
    mean_score: float
    support: int


def _progress(callback: _PROGRESS | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_tools_dir() -> Path:
    configured = os.environ.get("OPENMATTE_V5_TOOLS")
    if configured:
        return Path(configured).expanduser().resolve()
    runtime_root = os.environ.get("OPENMATTE_V5_RUNTIME")
    if runtime_root:
        candidate = Path(runtime_root).expanduser() / "tools" / "openmatte_hdr"
        if candidate.is_dir():
            return candidate.resolve()
    return (_workspace_root() / "tools" / "openmatte_hdr").resolve()


def _runtime_bridge_path() -> Path:
    configured = os.environ.get("OPENMATTE_V5_BRIDGE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    runtime_root = os.environ.get("OPENMATTE_V5_RUNTIME")
    if runtime_root:
        candidate = (
            Path(runtime_root).expanduser()
            / "dev"
            / "v05-native"
            / "v5_gpu_bridge.dll"
        )
        if candidate.is_file():
            return candidate.resolve()
    return (_workspace_root() / "dev" / "v05-native" / "v5_gpu_bridge.dll").resolve()


def _import_native_module(tools_dir: Path) -> Any:
    """Import the sibling-layout native module from the resolved deployment."""
    existing = sys.modules.get("v05_gpu_native")
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path and Path(existing_path).resolve().parent != tools_dir:
            raise GpuFastSyncUnavailable(
                "GPU Fast Auto Sync found v05_gpu_native from an unexpected "
                f"runtime directory: {existing_path}; expected {tools_dir}"
            )
        return existing

    previous_path = list(sys.path)
    sys.path.insert(0, str(tools_dir))
    try:
        module = importlib.import_module("v05_gpu_native")
    except Exception as exc:  # pragma: no cover - deployment-specific
        raise GpuFastSyncUnavailable(
            f"GPU Fast Auto Sync native module import failed from {tools_dir}: {exc}"
        ) from exc
    finally:
        sys.path[:] = previous_path

    module_path = getattr(module, "__file__", None)
    if not module_path or Path(module_path).resolve().parent != tools_dir:
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync imported a native module outside the resolved "
            f"runtime directory: {module_path!r}"
        )
    return module


def _require_ascii_kernel_toolchain(cupy: Any) -> None:
    """Reject installations whose NVRTC include path cannot be opened.

    NVRTC on Windows resolves ``-I`` directories through the ANSI code page, so
    a CuPy installation below a non-ASCII directory makes every kernel that
    includes a CuPy header fail with a missing ``cupy/complex.cuh``.  Detecting
    that here keeps the GPU-only contract explicit instead of surfacing an
    unrelated compilation error deep inside the analysis.
    """
    # Use the unresolved module path: that is the include directory CuPy hands
    # to NVRTC, so a junction or symlink under an ASCII path is a valid fix.
    include_dir = Path(cupy.__file__).parent / "_core" / "include"
    try:
        str(include_dir).encode("ascii")
    except UnicodeEncodeError:
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync cannot compile CUDA kernels because the CuPy "
            f"include path is not ASCII: {include_dir}. NVRTC cannot open such "
            "paths on Windows; install or link the environment under an ASCII "
            "path. No CPU fallback is used."
        ) from None


def _resolve_runtime(device_index: int) -> _GpuRuntime:
    """Resolve all GPU-only dependencies and fail closed when incomplete."""
    tools_dir = _runtime_tools_dir()
    bridge_path = _runtime_bridge_path()
    required = (
        tools_dir / "v05_gpu_native.py",
        tools_dir / "v05_streaming.py",
        tools_dir / "resample_backend.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if not bridge_path.is_file():
        missing.append(str(bridge_path))
    if missing:
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync runtime is unavailable; missing: "
            + ", ".join(missing)
            + ". Install/deploy the V5 native runtime; no CPU fallback is used."
        )

    try:
        cupy = importlib.import_module("cupy")
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"CUDA device {device_index} unavailable (device count={device_count})"
            )
        cupy.cuda.Device(device_index).use()
    except Exception as exc:  # pragma: no cover - hardware-specific
        raise GpuFastSyncUnavailable(
            f"GPU Fast Auto Sync requires a usable CuPy/CUDA device: {exc}"
        ) from exc

    _require_ascii_kernel_toolchain(cupy)

    try:
        module = cupy.RawModule(code=_KERNEL_SOURCE, options=("--std=c++17",))
        prepare_kernel = module.get_function("openmatte_prepare_planes")
        moments_kernel = module.get_function("openmatte_plane_moments")
        descriptor_kernel = module.get_function("openmatte_descriptor")
    except Exception as exc:  # pragma: no cover - toolchain-specific
        raise GpuFastSyncUnavailable(
            f"GPU Fast Auto Sync could not build its CUDA descriptor kernels: {exc}"
        ) from exc

    try:
        native = _import_native_module(tools_dir)
    except GpuFastSyncUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - deployment-specific
        raise GpuFastSyncUnavailable(
            f"GPU Fast Auto Sync native runtime setup failed: {exc}"
        ) from exc

    try:
        ffmpeg_bin = get_media_tool_config().require("ffmpeg").parent
    except Exception as exc:
        raise GpuFastSyncUnavailable(
            f"GPU Fast Auto Sync requires FFmpeg beside the native decoder: {exc}"
        ) from exc
    return _GpuRuntime(
        cupy=cupy,
        native=native,
        ffmpeg_bin=Path(ffmpeg_bin),
        bridge_path=bridge_path,
        tools_dir=tools_dir,
        device_index=int(device_index),
        prepare_kernel=prepare_kernel,
        moments_kernel=moments_kernel,
        descriptor_kernel=descriptor_kernel,
    )


def _fps_fraction(source: SourceInfo) -> tuple[int, int]:
    stream = source.selected_stream
    if stream is None or float(stream.fps or 0.0) <= 0:
        raise SynchronizationError("GPU Fast Auto Sync source has invalid FPS")
    rational = str(getattr(stream, "fps_rational", "") or "")
    try:
        value = Fraction(rational) if rational else Fraction(str(float(stream.fps)))
    except (ValueError, ZeroDivisionError):
        value = Fraction(float(stream.fps)).limit_denominator(1_000_000)
    if value <= 0:
        raise SynchronizationError("GPU Fast Auto Sync source FPS must be positive")
    return int(value.numerator), int(value.denominator)


def _validate_sources(hdr_source: SourceInfo, om_source: SourceInfo) -> tuple[float, float]:
    hdr_stream = hdr_source.selected_stream
    om_stream = om_source.selected_stream
    if hdr_stream is None or om_stream is None:
        raise SynchronizationError("GPU Fast Auto Sync requires selected HDR and OM video streams")
    for label, stream in (("HDR", hdr_stream), ("OM", om_stream)):
        rate_type = str(getattr(getattr(stream, "frame_rate_type", ""), "value", ""))
        if rate_type.upper() == "VFR":
            raise SynchronizationError(
                f"GPU Fast Auto Sync requires CFR frame mapping; {label} is VFR. "
                "Use the reference Auto Sync for timestamp-based media."
            )
    hdr_fps = float(hdr_stream.fps or 0.0)
    om_fps = float(om_stream.fps or 0.0)
    if hdr_fps <= 0 or om_fps <= 0:
        raise SynchronizationError(
            f"GPU Fast Auto Sync requires positive FPS: HDR={hdr_fps}, OM={om_fps}"
        )
    if abs(hdr_fps - om_fps) > max(0.01, hdr_fps * 0.0005):
        raise SynchronizationError(
            "GPU Fast Auto Sync requires matching CFR frame rates for the frame-offset "
            f"contract: HDR={hdr_fps:.6f}, OM={om_fps:.6f}"
        )
    return hdr_fps, om_fps


def _source_duration(source: SourceInfo) -> float:
    stream = source.selected_stream
    if stream is None:
        return 0.0
    duration = float(stream.duration_seconds or 0.0)
    if duration > 0:
        return duration
    fps = float(stream.fps or 0.0)
    count = int(stream.frame_count or 0)
    return count / fps if fps > 0 and count > 0 else 0.0


def _source_frame_count(source: SourceInfo, fps: float, limit_seconds: float) -> int:
    stream = source.selected_stream
    count = int(stream.frame_count or 0) if stream is not None else 0
    if count <= 0:
        count = int(round(_source_duration(source) * fps))
    return max(1, min(count, int(math.ceil(limit_seconds * fps))))


def _decoder_name(source: SourceInfo, *, is_hdr: bool) -> str:
    codec = str(getattr(source.selected_stream, "codec", "") or "").lower()
    if "hevc" in codec or "h265" in codec:
        return "hevc_cuvid"
    if "h264" in codec or "avc" in codec:
        return "h264_cuvid"
    # Keep the existing V5 defaults for metadata that omits codec details.
    return "hevc_cuvid" if is_hdr else "h264_cuvid"


def _source_dimensions(source: SourceInfo) -> tuple[int, int]:
    stream = source.selected_stream
    width = int(getattr(stream, "width", 0) or 0)
    height = int(getattr(stream, "height", 0) or 0)
    if width <= 0 or height <= 0:
        raise SynchronizationError("GPU Fast Auto Sync source has invalid dimensions")
    return width, height


def _working_plane(hdr_source: SourceInfo) -> tuple[int, int]:
    """Reference working plane, derived from the HDR framing.

    The HDR frame defines the shared field of view because it is the theatrical
    crop of the same image the Open Matte source shows in full.
    """
    width, height = _source_dimensions(hdr_source)
    plane_width = max(_GRID_WIDTH, min(_DESCRIPTOR_WIDTH, width))
    plane_height = max(_GRID_HEIGHT, int(round(height * plane_width / width)))
    if plane_height % 2:
        plane_height -= 1
    return int(plane_width), int(max(_GRID_HEIGHT, plane_height))


def _plane_geometry(
    frame_width: int,
    frame_height: int,
    working: tuple[int, int],
) -> _Geometry:
    """Map a decoded frame onto the shared working plane.

    HDR frames arrive already resampled by the bridge, so their factors are one.
    OM frames arrive at full geometry and are centre-cropped to the reference
    aspect before an integer box reduction.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise GpuFastSyncUnavailable(
            f"Native GPU decoder returned an invalid frame size: {frame_width}x{frame_height}"
        )
    plane_width, plane_height = (int(value) for value in working)
    factor_x = max(1, frame_width // plane_width)
    factor_y = factor_x
    crop_width = min(frame_width, plane_width * factor_x)
    crop_height = plane_height * factor_y
    if crop_height > frame_height:
        factor_y = max(1, frame_height // plane_height)
        crop_height = min(frame_height, plane_height * factor_y)
    width = max(_GRID_WIDTH, crop_width // factor_x)
    height = max(_GRID_HEIGHT, crop_height // factor_y)
    return _Geometry(
        width=int(width),
        height=int(height),
        crop_left=int(max(0, (frame_width - crop_width) // 2)),
        crop_top=int(max(0, (frame_height - crop_height) // 2)),
        factor_x=int(factor_x),
        factor_y=int(factor_y),
        pool_width=max(1, int(width) // _GRID_WIDTH),
        pool_height=max(1, int(height) // _GRID_HEIGHT),
    )


def _open_decoder(
    runtime: _GpuRuntime,
    source: SourceInfo,
    *,
    is_hdr: bool,
    start_frame: int,
    fps_fraction: tuple[int, int],
    working: tuple[int, int],
) -> Any:
    return runtime.native.NativeDecoder(
        Path(source.path),
        _decoder_name(source, is_hdr=is_hdr),
        runtime.native.HDR_MODE if is_hdr else runtime.native.SDR_MODE,
        int(max(0, start_frame)),
        int(fps_fraction[0]),
        int(fps_fraction[1]),
        ffmpeg_bin=runtime.ffmpeg_bin,
        bridge_path=runtime.bridge_path,
        device_index=runtime.device_index,
        ring_size=4,
        # Only the HDR path supports pre-PQ resampling inside the bridge; OM is
        # reduced by the descriptor kernels instead, still entirely on device.
        target_size=working if is_hdr else None,
    )


def _accumulate_descriptor(runtime: _GpuRuntime, frame: Any, bank: _Bank, row: int) -> None:
    """Run the three fused kernels for one native frame into one bank row."""
    cupy = runtime.cupy
    image = cupy.asarray(frame.array)
    if image.ndim != 3 or int(image.shape[2]) != 3:
        raise GpuFastSyncUnavailable(
            f"Native GPU decoder returned an unexpected RGB shape: {getattr(image, 'shape', None)}"
        )
    geometry = bank.geometry
    pixels = geometry.pixel_count
    threads = 256
    blocks = (pixels + threads - 1) // threads
    runtime.prepare_kernel(
        (max(1, blocks),),
        (threads,),
        (
            image,
            int(image.shape[1]),
            int(image.shape[0]),
            geometry.crop_left,
            geometry.crop_top,
            geometry.factor_x,
            geometry.factor_y,
            geometry.width,
            geometry.height,
            bank.log_luma,
            bank.chroma_rg,
            bank.chroma_gb,
        ),
    )
    bank.moments.fill(0)
    reduction_blocks = min(64, max(1, blocks))
    runtime.moments_kernel(
        (reduction_blocks,),
        (threads,),
        (bank.log_luma, bank.chroma_rg, bank.chroma_gb, pixels, bank.moments),
    )
    runtime.descriptor_kernel(
        (reduction_blocks,),
        (threads,),
        (
            bank.log_luma,
            bank.chroma_rg,
            bank.chroma_gb,
            geometry.width,
            geometry.height,
            geometry.pool_width,
            geometry.pool_height,
            bank.moments,
            bank.matrix[row],
        ),
    )
    if bank.planes is not None:
        # Keep the standardised plane for the frame-resolution correlation pass.
        # Mean and deviation stay in device memory; nothing is read back.
        count = cupy.float32(pixels)
        mean = bank.moments[0] / count
        variance = cupy.maximum(
            bank.moments[1] / count - mean * mean, cupy.float32(1e-12)
        )
        bank.planes[row] = (bank.log_luma - mean) / cupy.sqrt(variance)


def _decode_into_bank(
    runtime: _GpuRuntime,
    source: SourceInfo,
    bank_holder: dict[str, _Bank | None],
    *,
    is_hdr: bool,
    start_frame: int,
    count: int,
    fps_fraction: tuple[int, int],
    working: tuple[int, int],
    capacity: int,
    store_planes: bool = False,
    progress_callback: _PROGRESS | None = None,
    progress_label: str = "",
) -> list[int]:
    """Decode a contiguous run and accumulate one descriptor per frame.

    Returns the frame indices that produced a descriptor. The bank is created on
    first use so its geometry can follow the real decoded frame size.
    """
    if start_frame < 0 or count <= 0:
        return []
    cupy = runtime.cupy
    stream = cupy.cuda.get_current_stream()
    decoded: list[int] = []
    decoder = None
    try:
        decoder = _open_decoder(
            runtime,
            source,
            is_hdr=is_hdr,
            start_frame=int(start_frame),
            fps_fraction=fps_fraction,
            working=working,
        )
        for local_index in range(int(count)):
            frame = None
            try:
                frame = decoder.next(stream)
                if frame is None:
                    break
                bank = bank_holder["bank"]
                if bank is None:
                    image = cupy.asarray(frame.array)
                    bank = _Bank.create(
                        cupy,
                        _plane_geometry(
                            int(image.shape[1]), int(image.shape[0]), working
                        ),
                        capacity,
                        store_planes=store_planes,
                    )
                    bank_holder["bank"] = bank
                frame_index = int(start_frame + local_index)
                row = bank.reserve(frame_index)
                if row is not None:
                    _accumulate_descriptor(runtime, frame, bank, row)
                    decoded.append(frame_index)
            finally:
                if frame is not None:
                    frame.release(stream)
            if (
                progress_callback is not None
                and progress_label
                and local_index
                and local_index % _PROGRESS_FRAME_INTERVAL == 0
            ):
                _progress(
                    progress_callback,
                    f"{progress_label} {local_index}/{int(count)} frames",
                )
    finally:
        # Synchronize the consumer stream before the native handle is closed;
        # this makes the borrowed-slot ownership invariant explicit.
        try:
            stream.synchronize()
        finally:
            if decoder is not None:
                decoder.close()
    return decoded


def _descriptors_for(bank: _Bank, frames: list[int]) -> tuple[_Descriptors, list[int]]:
    """Normalize only the requested rows so the work stays proportional to use."""
    present = [frame for frame in frames if bank.row_of(frame) is not None]
    if not present:
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync requested descriptor rows that were never decoded"
        )
    cupy = bank.cupy
    rows = cupy.asarray([bank.row_of(frame) for frame in present], dtype=cupy.int32)
    return _Descriptors.from_raw(cupy, bank.matrix[rows], bank.geometry), present


def _slice_descriptors(descriptors: _Descriptors, start: int, stop: int) -> _Descriptors:
    """View a contiguous row range without copying device memory."""
    return _Descriptors(
        cupy=descriptors.cupy,
        luma=descriptors.luma[start:stop],
        chroma=descriptors.chroma[start:stop],
        gradient=descriptors.gradient[start:stop],
        spatial=descriptors.spatial[start:stop],
        edge=descriptors.edge[start:stop],
        contrast=descriptors.contrast[start:stop],
    )


def _pairwise_similarity(cupy: Any, left: _Descriptors, right: _Descriptors) -> Any:
    """Score aligned descriptor rows; a single right row broadcasts over left.

    The weights, the histogram-intersection basis and the 0.35/0.55 thresholds
    match the existing Fast Auto Sync contract. Every component is bounded by
    one, so a perfect match scores exactly one instead of saturating above it.
    """
    luma = cupy.minimum(left.luma, right.luma).sum(axis=1)
    chroma = cupy.minimum(left.chroma, right.chroma).sum(axis=1) * 0.5
    gradient = cupy.minimum(left.gradient, right.gradient).sum(axis=1)
    correlation = (left.spatial * right.spatial).sum(axis=1) / float(_POOLED_VALUES)
    spatial = cupy.clip((correlation + 1.0) * 0.5, 0.0, 1.0)
    ratio = (left.edge + cupy.float32(1e-6)) / (right.edge + cupy.float32(1e-6))
    edge = cupy.exp(-cupy.abs(cupy.log(ratio)))
    return cupy.clip(
        0.32 * luma + 0.22 * chroma + 0.20 * gradient + 0.16 * spatial + 0.10 * edge,
        0.0,
        1.0,
    )


def _plane_rows(bank: _Bank, frames: list[int]) -> tuple[Any, list[int]]:
    """Gather standardised planes for the requested frames."""
    if bank.planes is None:
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync requested planes from a descriptor-only bank"
        )
    present = [frame for frame in frames if bank.row_of(frame) is not None]
    if not present:
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync requested planes that were never decoded"
        )
    cupy = bank.cupy
    rows = cupy.asarray([bank.row_of(frame) for frame in present], dtype=cupy.int32)
    return bank.planes[rows], present


def _plane_correlation(cupy: Any, left: Any, right: Any) -> Any:
    """Normalised cross-correlation of standardised planes, mapped to [0, 1].

    Both operands are zero-mean unit-deviation planes, so the mean of their
    product is the Pearson correlation. This resolves single frames, which the
    pooled 9x16 descriptor grid cannot do inside one shot.
    """
    correlation = (left * right).mean(axis=1)
    return cupy.clip((correlation + 1.0) * 0.5, 0.0, 1.0)


def _top_candidates(
    cupy: Any,
    scores: Any,
    frame_indices: list[int],
    anchor_frame: int,
    *,
    top_k: int,
    max_offset_frames: int,
) -> list[dict[str, float | int]]:
    """Pick separated top-K offsets; only a bounded top table reaches the host."""
    total = int(scores.size)
    if total <= 0:
        return []
    pool = min(_CANDIDATE_POOL, total)
    indices = cupy.argpartition(-scores, pool - 1)[:pool]
    order = cupy.argsort(-scores[indices])
    indices = indices[order]
    host_indices = cupy.asnumpy(indices).tolist()
    host_scores = cupy.asnumpy(scores[indices]).tolist()

    candidates: list[dict[str, float | int]] = []
    for position, score in zip(host_indices, host_scores):
        frame_index = int(frame_indices[int(position)])
        offset = int(anchor_frame - frame_index)
        if abs(offset) > int(max_offset_frames):
            continue
        if float(score) < _GPU_MIN_SIMILARITY:
            continue
        if any(
            abs(offset - int(existing["offset"])) < _CANDIDATE_MIN_SEPARATION_FRAMES
            for existing in candidates
        ):
            continue
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "offset": offset,
                "score": float(score),
                "hdr_frame": frame_index,
            }
        )
        if len(candidates) >= int(top_k):
            break
    return candidates


def _cluster_votes(anchors: list[_Anchor]) -> list[_Cluster]:
    """Group independent anchor votes into offset clusters."""
    votes = [
        anchor for anchor in anchors
        if anchor.best_offset is not None and anchor.best_score >= _GPU_MIN_SIMILARITY
    ]
    clusters: list[list[_Anchor]] = []
    for anchor in sorted(votes, key=lambda item: -item.best_score):
        placed = False
        for cluster in clusters:
            center = cluster[0].best_offset or 0
            if abs(int(anchor.best_offset or 0) - int(center)) <= (
                _FAST_DIAGNOSTIC_MAX_AGREEMENT_TOLERANCE_FRAMES
            ):
                cluster.append(anchor)
                placed = True
                break
        if not placed:
            clusters.append([anchor])
    ranked: list[_Cluster] = []
    for cluster in clusters:
        # Anchors whose image actually changes between frames carry more weight,
        # because only they can resolve a frame-exact offset.
        weight = sum(item.best_score * (0.5 + item.resolvability) for item in cluster)
        offsets = [int(item.best_offset or 0) for item in cluster]
        ranked.append(
            _Cluster(
                offset=max(set(offsets), key=offsets.count),
                weight=float(weight),
                mean_score=float(sum(item.best_score for item in cluster) / len(cluster)),
                support=len(cluster),
            )
        )
    return sorted(ranked, key=lambda item: (-item.support, -item.weight, -item.mean_score))


def find_fast_global_offset_gpu(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    config: SyncConfig | None = None,
    *,
    progress_callback: _PROGRESS | None = None,
    device_index: int = 0,
) -> SyncModel:
    """Find a bounded GPU-only first-ten-minute synchronization proposal."""
    started = time.perf_counter()
    config = config or SyncConfig()
    hdr_fps, om_fps = _validate_sources(hdr_source, om_source)
    hdr_fps_fraction = _fps_fraction(hdr_source)
    om_fps_fraction = _fps_fraction(om_source)
    hdr_duration = _source_duration(hdr_source)
    om_duration = _source_duration(om_source)
    analysis_limit = min(GPU_FAST_MAX_SECONDS, hdr_duration, om_duration)
    if analysis_limit < 30.0:
        raise SynchronizationError(
            "GPU Fast Auto Sync requires at least 30 seconds in both sources; "
            f"range={analysis_limit:.2f}s"
        )

    runtime = _resolve_runtime(int(device_index))
    cupy = runtime.cupy
    working = _working_plane(hdr_source)
    hdr_frame_count = _source_frame_count(hdr_source, hdr_fps, analysis_limit)
    om_frame_count = _source_frame_count(om_source, om_fps, analysis_limit)
    limit_frames = min(hdr_frame_count, om_frame_count)
    max_offset_frames = int(round(config.search_range_seconds * hdr_fps))

    # The HDR scan band must stay inside the bounded window on both sides of
    # every anchor, so it is clamped against the analysed frame range.
    guard = _ANCHOR_WINDOW_FRAMES + _REFINE_HALF_FRAMES + 1
    band_frames = min(max_offset_frames, max(1, (limit_frames - 2 * guard) // 2))
    if band_frames < int(round(hdr_fps)):
        raise SynchronizationError(
            "GPU Fast Auto Sync has no usable bounded search band; "
            f"range={analysis_limit:.2f}s, band={band_frames} frames"
        )

    first_center = band_frames + guard
    last_center = limit_frames - band_frames - guard
    if last_center <= first_center:
        raise SynchronizationError("GPU Fast Auto Sync has no safe bounded sample interval")
    span = last_center - first_center
    divisor = max(1, _ANCHOR_COUNT - 1)
    anchors = [
        _Anchor(
            sample=index + 1,
            center_frame=int(first_center + round(span * index / divisor)),
            timestamp=float((first_center + span * index / divisor) / om_fps),
        )
        for index in range(_ANCHOR_COUNT)
    ]
    unique: dict[int, _Anchor] = {}
    for anchor in anchors:
        unique.setdefault(anchor.center_frame, anchor)
    anchors = list(unique.values())
    if len(anchors) < _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES:
        raise SynchronizationError("GPU Fast Auto Sync found too few bounded anchors")

    _progress(progress_callback, "Fast Auto Sync GPU: opening native NVDEC runtime")
    _progress(
        progress_callback,
        f"Fast Auto Sync GPU: sampling {len(anchors)} OM anchors "
        f"({_ANCHOR_WINDOW_FRAMES} frames each)",
    )
    om_holder: dict[str, _Bank | None] = {"bank": None}
    om_capacity = (len(anchors) + 1) * _ANCHOR_WINDOW_FRAMES
    # A single bootstrap OM window locates the matching shot; the anchors that
    # decide the frame-exact offset are chosen later, where the image moves.
    bootstrap = anchors[len(anchors) // 2]
    _decode_into_bank(
        runtime,
        om_source,
        om_holder,
        is_hdr=False,
        start_frame=max(0, bootstrap.center_frame - _REFINE_HALF_FRAMES),
        count=_ANCHOR_WINDOW_FRAMES,
        fps_fraction=om_fps_fraction,
        working=working,
        capacity=om_capacity,
        store_planes=True,
    )
    om_bank = om_holder["bank"]
    if om_bank is None or om_bank.row_of(bootstrap.center_frame) is None:
        raise SynchronizationError("GPU Fast Auto Sync found too few OM GPU descriptors")
    primary = bootstrap

    # One contiguous HDR run covers the whole search band for the primary
    # anchor. HDR frames are the cheap side because the bridge resamples them.
    band_start = max(0, primary.center_frame - band_frames)
    band_count = min(2 * band_frames + 1, hdr_frame_count - band_start)
    hdr_capacity = band_count
    hdr_holder: dict[str, _Bank | None] = {"bank": None}
    _progress(
        progress_callback,
        f"Fast Auto Sync GPU: scanning {band_count} HDR frames "
        f"(±{band_frames / hdr_fps:.1f}s around anchor {primary.sample})",
    )
    band_decoded = _decode_into_bank(
        runtime,
        hdr_source,
        hdr_holder,
        is_hdr=True,
        start_frame=band_start,
        count=band_count,
        fps_fraction=hdr_fps_fraction,
        working=working,
        capacity=hdr_capacity,
        progress_callback=progress_callback,
        progress_label="Fast Auto Sync GPU: HDR band",
    )
    hdr_bank = hdr_holder["bank"]
    if hdr_bank is None or len(band_decoded) < 3:
        raise SynchronizationError("GPU Fast Auto Sync found too few HDR GPU descriptors")
    if hdr_bank.geometry.width != om_bank.geometry.width or (
        hdr_bank.geometry.height != om_bank.geometry.height
    ):
        raise GpuFastSyncUnavailable(
            "GPU Fast Auto Sync produced mismatched working planes: "
            f"HDR={hdr_bank.geometry.width}x{hdr_bank.geometry.height}, "
            f"OM={om_bank.geometry.width}x{om_bank.geometry.height}"
        )

    _progress(progress_callback, "Fast Auto Sync GPU: device top-K offset scoring")
    band_descriptors, band_frames_present = _descriptors_for(hdr_bank, band_decoded)
    primary_descriptor, _ = _descriptors_for(om_bank, [primary.center_frame])
    band_scores = _pairwise_similarity(cupy, band_descriptors, primary_descriptor)
    candidates = _top_candidates(
        cupy,
        band_scores,
        band_frames_present,
        primary.center_frame,
        top_k=_GPU_TOP_K,
        max_offset_frames=max_offset_frames,
    )
    if not candidates:
        raise SynchronizationError("GPU Fast Auto Sync found no valid top-K offset candidates")

    # Frame-resolution pass. Each anchor correlates full standardised planes
    # inside a short window around every leading hypothesis, which resolves the
    # single frame that the pooled descriptor grid cannot separate.
    # One contiguous offset range covers every coarse hypothesis, so a peak can
    # never sit on a window boundary where the true maximum may lie outside.
    coarse_offsets = [int(candidate["offset"]) for candidate in candidates]
    fine_low = min(coarse_offsets) - _FINE_HALF_FRAMES
    fine_high = max(coarse_offsets) + _FINE_HALF_FRAMES
    if fine_high - fine_low + 1 > _FINE_MAX_FRAMES:
        center = int(candidates[0]["offset"])
        half = _FINE_MAX_FRAMES // 2
        fine_low, fine_high = center - half, center + half
    fine_low = max(fine_low, -max_offset_frames)
    fine_high = min(fine_high, max_offset_frames)
    fine_window = fine_high - fine_low + 1
    _progress(
        progress_callback,
        "Fast Auto Sync GPU: frame-resolution plane correlation over offsets "
        f"[{fine_low}, {fine_high}]",
    )
    # Anchor selection. A frame-exact offset is only observable where the image
    # changes between neighbouring frames, so the anchors are placed at the
    # strongest temporal transitions of the already decoded HDR band.
    # Anchors live on the HDR side, where the band scan already told us which
    # frames move. Mapping them onto OM through the coarse offset would land
    # tens of frames away, often inside a static shot, so the OM side is the
    # one that gets searched.
    band_length = len(band_frames_present)
    if band_length < 3:
        raise SynchronizationError("GPU Fast Auto Sync band is too short to place anchors")
    band_characteristic = cupy.asnumpy(band_descriptors.characteristic_scores()).tolist()
    temporal_change = 1.0 - _pairwise_similarity(
        cupy,
        _slice_descriptors(band_descriptors, 0, band_length - 1),
        _slice_descriptors(band_descriptors, 1, band_length),
    )
    host_change = cupy.asnumpy(temporal_change).tolist()
    # A single spike is a shot cut: one side of the correlation curve stays
    # flat, so the exact frame remains ambiguous. Sustained change means real
    # motion, which is what makes a frame-exact peak observable.
    motion = [
        min(
            host_change[max(0, position - 1)],
            host_change[position],
            host_change[min(len(host_change) - 1, position + 1)],
        )
        for position in range(len(host_change))
    ]

    def _anchor_fits(hdr_frame: int) -> bool:
        return (
            hdr_frame - _REFINE_HALF_FRAMES >= 0
            and hdr_frame + _REFINE_HALF_FRAMES < hdr_frame_count
            and hdr_frame + fine_low >= 0
            and hdr_frame + fine_high < om_frame_count
        )

    eligible = [
        position
        for position in range(band_length - 1)
        if _anchor_fits(int(band_frames_present[position]))
    ]
    if len(eligible) < _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES:
        raise SynchronizationError(
            "GPU Fast Auto Sync found no bounded window for frame-resolution anchors"
        )
    segment = max(1, len(eligible) // _ANCHOR_COUNT)
    selected: list[int] = []
    for index in range(_ANCHOR_COUNT):
        chunk = eligible[index * segment:(index + 1) * segment]
        if not chunk:
            continue
        selected.append(max(chunk, key=lambda position: motion[position]))
    anchors = []
    for sample, position in enumerate(sorted(set(selected)), start=1):
        hdr_frame = int(band_frames_present[position])
        anchors.append(
            _Anchor(
                sample=sample,
                center_frame=hdr_frame,
                timestamp=float(hdr_frame / hdr_fps),
                characteristic=float(band_characteristic[position]),
                resolvability=float(motion[position]),
            )
        )
    if len(anchors) < _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES:
        raise SynchronizationError("GPU Fast Auto Sync could not place enough anchors")
    _progress(
        progress_callback,
        f"Fast Auto Sync GPU: correlating {len(anchors)} moving anchors against "
        f"{fine_window}-frame OM windows",
    )

    hdr_fine_capacity = len(anchors) * _ANCHOR_WINDOW_FRAMES
    om_fine_capacity = len(anchors) * fine_window
    hdr_fine_holder: dict[str, _Bank | None] = {"bank": None}
    om_fine_holder: dict[str, _Bank | None] = {"bank": None}
    fine_records: list[dict[str, Any]] = []
    narrowed_offset: int | None = None
    for anchor in sorted(anchors, key=lambda item: -item.resolvability):
        _decode_into_bank(
            runtime,
            hdr_source,
            hdr_fine_holder,
            is_hdr=True,
            start_frame=anchor.center_frame - _REFINE_HALF_FRAMES,
            count=_ANCHOR_WINDOW_FRAMES,
            fps_fraction=hdr_fps_fraction,
            working=working,
            capacity=hdr_fine_capacity,
            store_planes=True,
        )
        hdr_fine_bank = hdr_fine_holder["bank"]
        if hdr_fine_bank is None or hdr_fine_bank.row_of(anchor.center_frame) is None:
            continue
        anchor_plane, _ = _plane_rows(hdr_fine_bank, [anchor.center_frame])
        # Once one anchor has resolved a sharp offset, the remaining anchors only
        # need a narrow window around it, which is the dominant cost saving.
        search_low, search_high = fine_low, fine_high
        if narrowed_offset is not None:
            search_low = max(fine_low, narrowed_offset - _FINE_HALF_FRAMES)
            search_high = min(fine_high, narrowed_offset + _FINE_HALF_FRAMES)
        window_start = anchor.center_frame + search_low
        window_count = search_high - search_low + 1
        if window_start < 0 or window_start + window_count > om_frame_count:
            continue
        _decode_into_bank(
            runtime,
            om_source,
            om_fine_holder,
            is_hdr=False,
            start_frame=window_start,
            count=window_count,
            fps_fraction=om_fps_fraction,
            working=working,
            capacity=om_fine_capacity,
            store_planes=True,
        )
        fine_bank = om_fine_holder["bank"]
        if fine_bank is None:
            continue
        requested = list(range(window_start, window_start + window_count))
        if not any(fine_bank.row_of(frame) is not None for frame in requested):
            continue
        planes, window_frames = _plane_rows(fine_bank, requested)
        host_scores = cupy.asnumpy(_plane_correlation(cupy, planes, anchor_plane)).tolist()
        peak = max(range(len(host_scores)), key=lambda index: host_scores[index])
        sidelobe_positions = [
            position
            for position in range(len(host_scores))
            if abs(position - peak) > _FINE_PEAK_GUARD_FRAMES
        ]
        sidelobe_peak = (
            max(sidelobe_positions, key=lambda index: host_scores[index])
            if sidelobe_positions
            else None
        )
        neighbourhood = [
            host_scores[position]
            for position in range(len(host_scores))
            if 1 <= abs(position - peak) <= _FINE_NEIGHBOURHOOD_FRAMES
        ]
        sharpness = (
            float(host_scores[peak] - max(neighbourhood)) if neighbourhood else 0.0
        )
        fine_records.append(
            {
                "anchor": anchor,
                "offset": int(window_frames[peak] - anchor.center_frame),
                "score": float(host_scores[peak]),
                "sharpness": sharpness,
                "resolved": bool(sharpness >= _FINE_MIN_PEAK_SHARPNESS),
                "sidelobe": (
                    float(host_scores[sidelobe_peak]) if sidelobe_peak is not None else 0.0
                ),
                "sidelobe_offset": (
                    int(window_frames[sidelobe_peak] - anchor.center_frame)
                    if sidelobe_peak is not None
                    else None
                ),
                "peak_position": int(peak),
                "window_frames": len(host_scores),
                "boundary_peak": bool(
                    peak <= _FINE_PEAK_GUARD_FRAMES
                    or peak >= len(host_scores) - 1 - _FINE_PEAK_GUARD_FRAMES
                ),
                "searched_offsets": [int(search_low), int(search_high)],
            }
        )
        if narrowed_offset is None and sharpness >= _FINE_MIN_PEAK_SHARPNESS:
            narrowed_offset = int(window_frames[peak] - anchor.center_frame)
    if not fine_records:
        raise SynchronizationError(
            "GPU Fast Auto Sync could not correlate any frame-resolution window"
        )
    resolved_records = [record for record in fine_records if bool(record["resolved"])]
    # With no sharp peak at all the material is static in every window; the
    # proposal is still returned, but its spread and margin report the weakness.
    winning_records = resolved_records or fine_records
    for record in winning_records:
        anchor = record["anchor"]
        anchor.best_offset = int(record["offset"])
        anchor.best_score = float(record["score"])
        anchor.source_candidate = int(candidates[0]["offset"])

    clusters = _cluster_votes(anchors)
    if not clusters:
        raise SynchronizationError(
            "GPU Fast Auto Sync produced no in-range consensus candidates"
        )
    best_offset = int(clusters[0].offset)
    if abs(best_offset) > max_offset_frames:
        raise SynchronizationError(
            "GPU Fast Auto Sync candidates are outside the configured search range"
        )

    # Consecutive-frame verification reuses descriptors that are already on the
    # device, so it costs no extra decode work.
    _progress(progress_callback, "Fast Auto Sync GPU: final consecutive-frame verification")
    verification_records: list[dict[str, float | int | bool]] = []
    verification_scores: list[float] = []
    verification_valid_frame_samples = 0
    for anchor in sorted(anchors, key=lambda item: -item.resolvability)[:_GPU_VERIFY_ANCHORS]:
        pair_frames_hdr: list[int] = []
        pair_frames_om: list[int] = []
        hdr_fine_bank = hdr_fine_holder["bank"]
        fine_bank = om_fine_holder["bank"]
        for step in range(-_REFINE_HALF_FRAMES, _REFINE_HALF_FRAMES + 1):
            hdr_frame = anchor.center_frame + step
            om_frame = hdr_frame + best_offset
            if hdr_fine_bank is None or fine_bank is None:
                continue
            if hdr_fine_bank.row_of(hdr_frame) is None:
                continue
            if fine_bank.row_of(om_frame) is None:
                continue
            pair_frames_om.append(om_frame)
            pair_frames_hdr.append(hdr_frame)
        if not pair_frames_hdr:
            verification_records.append(
                {
                    "sample": anchor.sample,
                    "score": 0.0,
                    "frame_samples": 0,
                    "valid_frame_samples": 0,
                    "independent": False,
                }
            )
            continue
        if fine_bank is None or hdr_fine_bank is None:
            continue
        hdr_pairs, _ = _plane_rows(hdr_fine_bank, pair_frames_hdr)
        om_pairs, _ = _plane_rows(fine_bank, pair_frames_om)
        host_pairs = cupy.asnumpy(_plane_correlation(cupy, hdr_pairs, om_pairs)).tolist()
        valid = sum(1 for score in host_pairs if score >= _GPU_VERIFY_SIMILARITY)
        mean_score = float(sum(host_pairs) / len(host_pairs))
        verification_valid_frame_samples += int(valid)
        verification_scores.append(mean_score)
        verification_records.append(
            {
                "sample": anchor.sample,
                "hdr_frame": int(anchor.center_frame),
                "om_frame": int(anchor.center_frame + best_offset),
                "score": mean_score,
                "frame_samples": len(host_pairs),
                "valid_frame_samples": int(valid),
                "independent": bool(valid >= max(3, len(host_pairs) // 2)),
            }
        )
    post_score = (
        float(sum(verification_scores) / len(verification_scores))
        if verification_scores
        else 0.0
    )
    verification_independent_count = sum(
        1 for record in verification_records if bool(record["independent"])
    )
    post_agreement = (
        verification_independent_count / len(verification_records)
        if verification_records
        else 0.0
    )

    independent_offsets = [
        int(anchor.best_offset)
        for anchor in anchors
        if anchor.best_offset is not None and anchor.best_score >= _GPU_MIN_SIMILARITY
    ]
    independent_sample_count = len(independent_offsets)
    agreement_count = sum(
        1
        for offset in independent_offsets
        if abs(offset - best_offset) <= _FAST_DIAGNOSTIC_MAX_AGREEMENT_TOLERANCE_FRAMES
    )
    agreement_percentage = (
        agreement_count / independent_sample_count if independent_sample_count else 0.0
    )
    offset_spread_frames = (
        float(max(independent_offsets) - min(independent_offsets))
        if independent_offsets
        else None
    )

    # Scores and margin come from the frame-resolution pass: the peak against
    # the strongest competing frame outside the peak neighbourhood.
    best_candidate_score = float(
        sum(record["score"] for record in winning_records) / len(winning_records)
    )
    second_candidate_score = float(
        sum(record["sidelobe"] for record in winning_records) / len(winning_records)
    )
    score_margin = best_candidate_score - second_candidate_score
    sidelobe_offsets = [
        int(record["sidelobe_offset"])
        for record in winning_records
        if record["sidelobe_offset"] is not None
    ]
    second_best_offset = (
        max(set(sidelobe_offsets), key=sidelobe_offsets.count) if sidelobe_offsets else None
    )
    total_weight = sum(cluster.weight for cluster in clusters)
    consensus = clusters[0].weight / total_weight if total_weight > 0 else 0.0
    margin_quality = max(0.0, min(1.0, score_margin / _FAST_DIAGNOSTIC_MIN_MARGIN))
    spread_quality = (
        1.0
        if offset_spread_frames is not None
        and offset_spread_frames <= _FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES
        else 0.0
    )
    verification_coverage = max(
        0.0,
        min(1.0, verification_independent_count / _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES),
    )
    fast_confidence = max(
        0.0,
        min(
            1.0,
            0.10 * consensus
            + 0.10 * best_candidate_score
            + 0.15 * margin_quality
            + 0.25 * agreement_percentage
            + 0.10 * spread_quality
            + 0.20 * post_score
            + 0.10 * verification_coverage,
        ),
    )
    confidence = max(
        0.0,
        min(
            1.0,
            0.25 * consensus
            + 0.25 * clusters[0].mean_score
            + 0.35 * post_score
            + 0.15 * post_agreement,
        ),
    )
    status = SyncStatus.LOCKED if confidence >= config.min_confidence else SyncStatus.FAILED
    frame_locked = status == SyncStatus.LOCKED
    fast_diagnostic_status = (
        "FAST_LOCKED"
        if (
            fast_confidence >= _FAST_DIAGNOSTIC_MIN_CONFIDENCE
            and independent_sample_count >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES
            and agreement_count >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES
            and agreement_percentage == 1.0
            and offset_spread_frames is not None
            and offset_spread_frames <= _FAST_DIAGNOSTIC_MAX_SPREAD_FRAMES
            and score_margin >= _FAST_DIAGNOSTIC_MIN_MARGIN
            and verification_independent_count >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES
            and verification_valid_frame_samples
            >= _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES * _GPU_VERIFY_WINDOW_FRAMES
        )
        else "FAST_REVIEW"
    )
    candidate_rank = next(
        (
            int(candidate["rank"])
            for candidate in candidates
            if int(candidate["offset"]) == best_offset
        ),
        None,
    )
    elapsed = time.perf_counter() - started
    diagnostics = {
        "schema_version": 3,
        "strategy": "gpu_native_hdr_band_scan_anchor_consensus",
        "gpu_only": True,
        "analysis_limit_seconds": float(analysis_limit),
        "elapsed_seconds": float(elapsed),
        "device_index": int(runtime.device_index),
        "runtime_tools_dir": str(runtime.tools_dir),
        "bridge_path": str(runtime.bridge_path),
        "full_frame_d2h": False,
        "host_transfer_policy": "scalar_reductions_and_top_k_only",
        "candidate_score_basis": "gpu_fused_kernel_log_standardised_descriptors",
        "best_offset": int(best_offset),
        "best_score": float(best_candidate_score),
        "second_best_offset": second_best_offset,
        "second_score": float(second_candidate_score),
        "score_margin": float(score_margin),
        "selected_offset": int(best_offset),
        "selected_candidate_rank": candidate_rank,
        "candidates": [
            {
                "rank": int(candidate["rank"]),
                "offset": int(candidate["offset"]),
                "score": float(candidate["score"]),
                "hdr_frame": int(candidate["hdr_frame"]),
            }
            for candidate in candidates
        ],
        "anchors": [
            {
                "sample": anchor.sample,
                "hdr_frame": int(anchor.center_frame),
                "om_frame": (
                    int(anchor.center_frame + anchor.best_offset)
                    if anchor.best_offset is not None
                    else None
                ),
                "timestamp": float(anchor.timestamp),
                "characteristic_score": float(anchor.characteristic),
                "temporal_resolvability": float(anchor.resolvability),
                "best_offset": anchor.best_offset,
                "best_score": float(anchor.best_score),
                "independent": bool(
                    anchor.best_offset is not None
                    and anchor.best_score >= _GPU_MIN_SIMILARITY
                ),
                "is_primary": anchor is primary,
            }
            for anchor in anchors
        ],
        "agreement_count": int(agreement_count),
        "agreement_percentage": float(agreement_percentage),
        "offset_spread_frames": offset_spread_frames,
        "independent_sample_count": int(independent_sample_count),
        "minimum_independent_samples": _FAST_DIAGNOSTIC_MIN_INDEPENDENT_SAMPLES,
        "verification_anchors": verification_records,
        "verification": {
            "consecutive": True,
            "gpu": True,
            "window_frames": _GPU_VERIFY_WINDOW_FRAMES,
            "anchor_limit": _GPU_VERIFY_ANCHORS,
            "similarity_threshold": _GPU_VERIFY_SIMILARITY,
            "post_selection_score": float(post_score),
            "post_selection_similarity_agreement": float(post_agreement),
            "post_selection_independent_anchors": int(verification_independent_count),
            "post_selection_valid_frame_samples": int(verification_valid_frame_samples),
            "reused_device_descriptors": True,
        },
        "sampling": {
            "anchor_count": len(anchors),
            "primary_anchor": primary.sample,
            "om_bootstrap_descriptors": int(om_bank.count),
            "hdr_band_descriptors": int(hdr_bank.count),
            "hdr_band_frames": int(band_count),
            "hdr_band_seconds": float(band_count / hdr_fps),
            "anchor_window_frames": _ANCHOR_WINDOW_FRAMES,
            "consensus_candidates": _CONSENSUS_CANDIDATES,
            "fine_offset_range": [int(fine_low), int(fine_high)],
            "fine_window_frames": int(fine_window),
            "fine_boundary_peaks": sum(
                1 for record in fine_records if bool(record["boundary_peak"])
            ),
            "fine_resolved_anchors": len(resolved_records),
            "fine_peak_sharpness": [
                round(float(record["sharpness"]), 5) for record in fine_records
            ],
            "min_peak_sharpness": _FINE_MIN_PEAK_SHARPNESS,
            "fine_om_descriptors": int(
                om_fine_holder["bank"].count if om_fine_holder["bank"] is not None else 0
            ),
            "fine_hdr_descriptors": int(
                hdr_fine_holder["bank"].count if hdr_fine_holder["bank"] is not None else 0
            ),
            "fine_stage": "standardised_plane_normalised_cross_correlation",
            "max_offset_frames": int(max_offset_frames),
            "working_plane": [hdr_bank.geometry.width, hdr_bank.geometry.height],
            "om_device_downscale_factor": om_bank.geometry.factor_x,
            "om_field_of_view_crop": [
                om_bank.geometry.crop_left,
                om_bank.geometry.crop_top,
            ],
            "descriptor_domain": "log_luminance_standardised_per_frame",
        },
        "fast_confidence_components": {
            "candidate_consensus": float(consensus),
            "best_candidate_score": float(best_candidate_score),
            "margin_quality": float(margin_quality),
            "sample_agreement": float(agreement_percentage),
            "spread_quality": float(spread_quality),
            "post_selection_verification": float(post_score),
            "verification_coverage": float(verification_coverage),
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
    _progress(
        progress_callback,
        f"Fast Auto Sync GPU: {status.value}, offset={best_offset}, "
        f"confidence={confidence:.3f}, elapsed={elapsed:.1f}s",
    )
    return SyncModel(
        frame_offset=int(best_offset),
        confidence=float(confidence),
        status=status,
        frame_locked=frame_locked,
        offset_seconds=float(best_offset / hdr_fps),
        method="fast_gpu_native_first_10m_band_scan",
        fast_confidence=float(fast_confidence),
        fast_diagnostic_status=fast_diagnostic_status,
        fast_diagnostics=diagnostics,
    )


__all__ = ["GpuFastSyncUnavailable", "find_fast_global_offset_gpu"]
