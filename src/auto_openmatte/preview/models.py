"""Backend-neutral data models for single-frame preview."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from auto_openmatte.backends import Frame, MemoryDomain


class PreviewMode(str, Enum):
    """Single-frame preview operation."""

    SOURCE_HDR = "SOURCE_HDR"
    SOURCE_OM = "SOURCE_OM"
    CORRECTED_OM = "CORRECTED_OM"


class PreviewQuality(str, Enum):
    """Quality/priority level, not a color algorithm."""

    DRAFT = "DRAFT"
    FULL = "FULL"


@dataclass(frozen=True)
class ResizeRequest:
    """Neutral GPU/host resize request with no fixed project resolution."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    method: str = "INTER_LINEAR"
    memory_domain: MemoryDomain = MemoryDomain.DEVICE

    def __post_init__(self) -> None:
        for name in (
            "source_width",
            "source_height",
            "target_width",
            "target_height",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.method:
            raise ValueError("resize method is required")
        object.__setattr__(self, "memory_domain", MemoryDomain(self.memory_domain))


@dataclass(frozen=True)
class PreviewRequest:
    """Immutable request carrying all state that affects a rendered frame."""

    frame_number: int
    locked_offset: int
    sync_version: str
    quality: PreviewQuality
    mode: PreviewMode
    resize: ResizeRequest
    shot_id: int | None = None
    model_version: str = ""
    settled: bool = False

    def __post_init__(self) -> None:
        if self.frame_number < 0:
            raise ValueError("frame_number must be non-negative")
        object.__setattr__(self, "quality", PreviewQuality(self.quality))
        object.__setattr__(self, "mode", PreviewMode(self.mode))

    @property
    def om_frame_number(self) -> int:
        """Deterministic frame mapping from the already locked offset."""
        return self.frame_number + self.locked_offset


@dataclass
class FramePair:
    """One synchronized pair owned until the renderer releases both frames."""

    hdr: Frame
    om: Frame
    hdr_frame_number: int
    om_frame_number: int
    locked_offset: int
    sync_version: str
    sequence: int
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        first_error: BaseException | None = None
        for frame in (self.hdr, self.om):
            try:
                frame.release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        self._released = True


@dataclass(frozen=True)
class PreviewFrame:
    """Small host-display payload and complete provenance metadata.

    ``rgb24`` is the only display boundary payload.  It is intentionally a
    byte string rather than a Qt, OpenCV, CuPy, CUDA-pointer, or AVFrame type.
    The backend may keep all full-resolution work in device memory until this
    final, bounded-size materialization step.
    """

    rgb24: bytes
    width: int
    height: int
    frame_number: int
    om_frame_number: int
    shot_id: int | None
    locked_offset: int
    sync_version: str
    quality: PreviewQuality
    source_role: PreviewMode
    renderer_id: str
    model_version: str
    backend_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preview dimensions must be positive")
        if len(self.rgb24) != self.width * self.height * 3:
            raise ValueError("rgb24 payload size does not match preview dimensions")


@dataclass(frozen=True)
class PreviewCapabilities:
    """Capability report independent from backend-native object types."""

    backend_id: str
    available: bool
    hardware_decode: bool
    gpu_resize: bool
    gpu_transform: bool
    gpu_preview: bool
    reason: str = ""


@dataclass(frozen=True)
class PreviewStatus:
    """Readiness/status exposed to GUI without exposing backend internals."""

    state: str
    message: str = ""
    request_generation: int = 0


@dataclass(frozen=True)
class PreviewCacheKey:
    """Versioned key separating source and corrected preview caches."""

    mode: PreviewMode
    quality: PreviewQuality
    frame_number: int
    om_frame_number: int
    shot_id: int | None
    sync_version: str
    model_version: str
    width: int
    height: int
    method: str
