"""GUI adapter for the backend-neutral single-frame preview engine."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from auto_openmatte.core.config import TransformConfig
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.preview import (
    CudaPreviewBackend,
    PreviewEngine,
    PreviewMode,
    PreviewQuality,
    PreviewRequest,
    ResizeRequest,
)
from auto_openmatte.preview.models import PreviewFrame
from gui.models.state import AppState


class PreviewController(QObject):
    """Own preview lifecycle without owning synchronization state.

    The controller reads a copy of the currently locked shot decision when a
    request starts.  It never edits proposals/locks and never uses the sync
    controller's playback or decoder state.
    """

    frame_ready = Signal(object)  # PreviewFrame
    error = Signal(str)
    status = Signal(str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.backend = CudaPreviewBackend()
        self.engine = PreviewEngine(self.backend)
        self._session_signature: tuple | None = None
        self._shot_transform: ShotTransform | None = None
        self._transform_config = TransformConfig()

    @property
    def capabilities(self):
        return self.backend.capabilities

    def set_shot_transform(self, transform: ShotTransform | None) -> None:
        """Set the already fitted transform supplied by the project layer."""
        self._shot_transform = transform
        self.engine.caches.invalidate_rendered()

    def set_transform_config(self, config: TransformConfig) -> None:
        """Use the production transform policy supplied by the project layer."""
        self._transform_config = config
        self.engine.caches.invalidate_rendered()

    def _locked_context(self) -> tuple[int, str, int] | None:
        """Return a read-only snapshot of the active locked shot decision."""
        active_frame = int(self.app_state.current_frame)
        lock = next(
            (
                item
                for item in self.app_state.shot_locks
                if item.hdr_start_frame <= active_frame <= item.hdr_end_frame
            ),
            None,
        )
        if lock is None and 0 <= self.app_state.current_shot_index < len(self.app_state.shot_locks):
            candidate = self.app_state.shot_locks[self.app_state.current_shot_index]
            if candidate.is_locked:
                lock = candidate
        if lock is None:
            lock = self.app_state.shot_locks[0] if self.app_state.shot_locks else None
        if lock is None:
            return None
        version = f"{lock.shot_id}:{lock.offset}:{lock.confidence:.6f}"
        return int(lock.offset), version, self.app_state.current_shot_index

    @staticmethod
    def _even_dimension(value: int) -> int:
        value = max(2, int(value))
        return value if value % 2 == 0 else value - 1

    def request_preview(
        self,
        mode: PreviewMode,
        quality: PreviewQuality,
        *,
        target_width: int,
        target_height: int,
    ) -> None:
        """Request one DRAFT/FULL frame through the GPU preview engine."""
        mode = PreviewMode(mode)
        quality = PreviewQuality(quality)
        hdr = self.app_state.hdr_source
        om = self.app_state.om_source
        if hdr is None or om is None or not hdr.path or not om.path:
            self._fail("Preview requires both HDR and OpenMatte sources")
            return

        locked = self._locked_context()
        if locked is None:
            self._fail("Preview requires an existing locked shot offset")
            return
        offset, sync_version, shot_index = locked

        if not self.capabilities.available:
            self._fail(
                "CUDA preview unavailable: "
                f"{self.capabilities.reason}"
            )
            return
        if mode == PreviewMode.CORRECTED_OM and self._shot_transform is None:
            self._fail("CORRECTED_OM preview requires the fitted ShotTransform")
            return

        source = hdr if mode == PreviewMode.SOURCE_HDR else om
        source_width = int(source.width)
        source_height = int(source.height)
        if source_width <= 0 or source_height <= 0:
            self._fail("Preview source dimensions are not available")
            return
        width = self._even_dimension(target_width)
        height = self._even_dimension(target_height)
        if width <= 0 or height <= 0:
            self._fail("Preview target dimensions must be positive")
            return

        signature = (
            hdr.path,
            om.path,
            offset,
            sync_version,
        )
        try:
            if signature != self._session_signature:
                self.engine.open(
                    Path(hdr.path),
                    Path(om.path),
                    locked_offset=offset,
                    sync_version=sync_version,
                )
                self._session_signature = signature

            model_version = ""
            if self._shot_transform is not None:
                model_version = hashlib.sha256(
                    repr(self._shot_transform).encode("utf-8")
                ).hexdigest()[:16]
            request = PreviewRequest(
                frame_number=int(self.app_state.current_frame),
                locked_offset=offset,
                sync_version=sync_version,
                quality=quality,
                mode=mode,
                resize=ResizeRequest(
                    source_width=source_width,
                    source_height=source_height,
                    target_width=width,
                    target_height=height,
                ),
                shot_id=shot_index,
                model_version=model_version,
                settled=quality == PreviewQuality.FULL,
            )
            self.status.emit(f"Preview {quality.value}: {mode.value}")
            self.engine.request_async(
                request,
                shot_transform=self._shot_transform,
                transform_config=self._transform_config,
                on_ready=self._on_frame_ready,
                on_error=self._on_engine_error,
            )
        except Exception as exc:
            self._fail(str(exc))

    def cancel(self) -> None:
        self.engine.cancel_current()
        self.status.emit("Preview cancelled")

    def _on_frame_ready(self, frame: PreviewFrame) -> None:
        self.frame_ready.emit(frame)

    def _on_engine_error(self, error: BaseException) -> None:
        self._fail(str(error))

    def _fail(self, message: str) -> None:
        self.status.emit(message)
        self.error.emit(message)

    def close(self) -> None:
        self.engine.close()
