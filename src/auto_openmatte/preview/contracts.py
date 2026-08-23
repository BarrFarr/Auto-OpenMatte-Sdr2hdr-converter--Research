"""Neutral contracts for the single-frame preview engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from auto_openmatte.backends import Frame
from auto_openmatte.core.config import TransformConfig
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.preview.models import (
    PreviewCapabilities,
    PreviewFrame,
    PreviewMode,
    PreviewRequest,
    ResizeRequest,
)


class PreviewCancelled(RuntimeError):
    """Raised when a superseded preview request stops at a safe boundary."""


class PreviewBackendUnavailable(RuntimeError):
    """Raised when the selected backend cannot be constructed in this checkout."""


class PreviewCancellation:
    """Small cancellation token shared by decoder, transform and resize stages."""

    def __init__(self) -> None:
        from threading import Event

        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise PreviewCancelled("preview request was cancelled")


class SingleSourceDecoderBackend(ABC):
    """Neutral lifecycle contract for one independently decoded source.

    Synchronization is deliberately outside this contract.  Callers provide
    the absolute frame number to ``read``; implementations must return one
    neutral :class:`Frame`, never a paired batch.
    """

    @property
    @abstractmethod
    def lifecycle(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def status(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def open(
        self,
        source: Path,
        decoder: str,
        start_frame: int,
        fps: tuple[int, int] | str | float,
    ) -> None:
        ...

    @abstractmethod
    def read(self, frame_number: int) -> Frame | None:
        ...

    @abstractmethod
    def seek(self, frame_number: int) -> None:
        ...

    @abstractmethod
    def cancel(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class PreviewDecoderSession(ABC):
    """Persistent session for one source role.

    Implementations own their decoder, bounded queue and bounded source cache.
    The contract exposes only neutral ``Frame`` objects and lifecycle methods.
    """

    @property
    @abstractmethod
    def status(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def seek(self, frame_number: int, cancellation: PreviewCancellation) -> None:
        ...

    @abstractmethod
    def read(self, frame_number: int, cancellation: PreviewCancellation) -> Frame:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def cancel(self) -> None:
        """Optional immediate cancellation hook for a backend decoder."""


class PreviewBackend(ABC):
    """Neutral preview operations implemented by a concrete device backend."""

    @property
    @abstractmethod
    def capabilities(self) -> PreviewCapabilities:
        ...

    @property
    @abstractmethod
    def backend_id(self) -> str:
        ...

    @abstractmethod
    def create_decoder_session(
        self,
        source_path: Path,
        *,
        source_role: PreviewMode,
        locked_offset: int,
    ) -> PreviewDecoderSession:
        ...

    @abstractmethod
    def resize(
        self,
        frame: Frame,
        request: ResizeRequest,
        cancellation: PreviewCancellation,
    ) -> Frame:
        ...

    @abstractmethod
    def transform(
        self,
        frame: Frame,
        shot_transform: ShotTransform,
        transform_config: TransformConfig,
        cancellation: PreviewCancellation,
    ) -> Frame:
        ...

    @abstractmethod
    def materialize(
        self,
        frame: Frame,
        request: PreviewRequest,
        source_role: PreviewMode,
        cancellation: PreviewCancellation,
    ) -> PreviewFrame:
        ...

    def close(self) -> None:
        """Release backend-level pools/resources when the engine closes."""


class SingleFrameRenderer:
    """Apply one selected preview mode through the supplied backend."""

    renderer_id = "single-frame-renderer-v1"

    def __init__(self, backend: PreviewBackend):
        self.backend = backend

    def render(
        self,
        pair,
        request: PreviewRequest,
        *,
        shot_transform: ShotTransform | None,
        transform_config: TransformConfig,
        cancellation: PreviewCancellation,
    ) -> PreviewFrame:
        cancellation.checkpoint()
        if request.mode == PreviewMode.SOURCE_HDR:
            selected = pair.hdr
            role = PreviewMode.SOURCE_HDR
        elif request.mode == PreviewMode.SOURCE_OM:
            selected = pair.om
            role = PreviewMode.SOURCE_OM
        elif request.mode == PreviewMode.CORRECTED_OM:
            if shot_transform is None:
                raise ValueError("CORRECTED_OM requires the fitted ShotTransform")
            selected = self.backend.transform(
                pair.om,
                shot_transform,
                transform_config,
                cancellation,
            )
            role = PreviewMode.CORRECTED_OM
        else:
            raise ValueError(f"unsupported preview mode: {request.mode}")

        cancellation.checkpoint()
        resized = self.backend.resize(selected, request.resize, cancellation)
        cancellation.checkpoint()
        return self.backend.materialize(resized, request, role, cancellation)
