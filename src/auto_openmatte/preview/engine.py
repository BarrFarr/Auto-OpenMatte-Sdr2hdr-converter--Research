"""Persistent single-frame preview engine and request scheduler."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Callable

from auto_openmatte.core.config import TransformConfig
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.preview.cache import PreviewCaches
from auto_openmatte.preview.contracts import (
    PreviewBackend,
    PreviewCancellation,
    PreviewCancelled,
    PreviewDecoderSession,
    SingleFrameRenderer,
)
from auto_openmatte.preview.models import (
    FramePair,
    PreviewCacheKey,
    PreviewFrame,
    PreviewMode,
    PreviewQuality,
    PreviewRequest,
    PreviewStatus,
)


class PreviewEngine:
    """Own two persistent source sessions and a single serialized render lane."""

    def __init__(
        self,
        backend: PreviewBackend,
        *,
        transform_config: TransformConfig | None = None,
        source_cache_entries: int = 6,
        rendered_cache_entries: int = 6,
    ) -> None:
        self.backend = backend
        self.transform_config = transform_config or TransformConfig()
        self.renderer = SingleFrameRenderer(backend)
        self.caches = PreviewCaches(source_cache_entries, rendered_cache_entries)
        self._hdr_session: PreviewDecoderSession | None = None
        self._om_session: PreviewDecoderSession | None = None
        self._hdr_path: Path | None = None
        self._om_path: Path | None = None
        self._locked_offset: int | None = None
        self._sync_version = ""
        self._status = PreviewStatus("closed")
        self._state_lock = RLock()
        self._generation = 0
        self._active_token: PreviewCancellation | None = None
        self._active_future: Future[PreviewFrame] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")

    @property
    def capabilities(self):
        return self.backend.capabilities

    @property
    def status(self) -> PreviewStatus:
        with self._state_lock:
            return self._status

    def open(
        self,
        hdr_path: Path,
        om_path: Path,
        *,
        locked_offset: int,
        sync_version: str,
    ) -> None:
        """Open HDR and OM sessions once for a locked synchronization version."""
        self.close_sessions()
        self.caches.invalidate_all()
        self._hdr_path = Path(hdr_path)
        self._om_path = Path(om_path)
        self._locked_offset = int(locked_offset)
        self._sync_version = str(sync_version)
        try:
            self._hdr_session = self.backend.create_decoder_session(
                self._hdr_path,
                source_role=PreviewMode.SOURCE_HDR,
                locked_offset=0,
            )
            self._om_session = self.backend.create_decoder_session(
                self._om_path,
                source_role=PreviewMode.SOURCE_OM,
                locked_offset=self._locked_offset,
            )
            self._hdr_session.open()
            self._om_session.open()
            with self._state_lock:
                self._status = PreviewStatus("ready", "persistent decoder sessions open")
        except BaseException:
            self.close_sessions()
            with self._state_lock:
                self._status = PreviewStatus("unavailable", "decoder session could not be opened")
            raise

    def close_sessions(self) -> None:
        for session in (self._hdr_session, self._om_session):
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
        self._hdr_session = None
        self._om_session = None

    def _cache_key(self, request: PreviewRequest) -> PreviewCacheKey:
        return PreviewCacheKey(
            mode=request.mode,
            quality=request.quality,
            frame_number=request.frame_number,
            om_frame_number=request.om_frame_number,
            shot_id=request.shot_id,
            sync_version=request.sync_version,
            model_version=request.model_version,
            width=request.resize.target_width,
            height=request.resize.target_height,
            method=request.resize.method,
        )

    def _get_pair(
        self,
        request: PreviewRequest,
        cancellation: PreviewCancellation,
    ) -> FramePair:
        if self._hdr_session is None or self._om_session is None:
            raise RuntimeError("preview decoder sessions are not open")
        if self._locked_offset != request.locked_offset:
            raise RuntimeError("preview request offset does not match the locked session")
        if self._sync_version != request.sync_version:
            raise RuntimeError("preview request sync version is stale")
        if request.om_frame_number < 0:
            raise ValueError("locked offset maps the requested frame before OM frame zero")

        cancellation.checkpoint()
        hdr = None
        om = None
        try:
            hdr = self._hdr_session.read(request.frame_number, cancellation)
            om = self._om_session.read(request.om_frame_number, cancellation)
            cancellation.checkpoint()
            return FramePair(
                hdr=hdr,
                om=om,
                hdr_frame_number=request.frame_number,
                om_frame_number=request.om_frame_number,
                locked_offset=request.locked_offset,
                sync_version=request.sync_version,
                sequence=request.frame_number,
            )
        except BaseException:
            for frame in (om, hdr):
                if frame is not None:
                    try:
                        frame.release()
                    except Exception:
                        pass
            raise

    def render_sync(
        self,
        request: PreviewRequest,
        *,
        shot_transform: ShotTransform | None = None,
        transform_config: TransformConfig | None = None,
        cancellation: PreviewCancellation | None = None,
    ) -> PreviewFrame:
        """Render one frame through the common decoder/transform/resize path."""
        token = cancellation or PreviewCancellation()
        key = self._cache_key(request)
        cache = self.caches.source if request.mode != PreviewMode.CORRECTED_OM else self.caches.rendered
        cached = cache.get(key)
        if cached is not None:
            return cached

        pair = self._get_pair(request, token)
        try:
            preview = self.renderer.render(
                pair,
                request,
                shot_transform=shot_transform,
                transform_config=transform_config or self.transform_config,
                cancellation=token,
            )
            token.checkpoint()
            cache.put(key, preview)
            return preview
        finally:
            pair.release()

    def request_async(
        self,
        request: PreviewRequest,
        *,
        shot_transform: ShotTransform | None = None,
        transform_config: TransformConfig | None = None,
        on_ready: Callable[[PreviewFrame], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> int:
        """Schedule one request; a newer DRAFT supersedes older work."""
        with self._state_lock:
            self._generation += 1
            generation = self._generation
            previous_token = self._active_token
            previous_future = self._active_future
            if previous_token is not None:
                previous_token.cancel()
            if previous_future is not None and (
                request.quality == PreviewQuality.DRAFT
                or request.quality == PreviewQuality.FULL
            ):
                previous_future.cancel()
            token = PreviewCancellation()
            self._active_token = token
            self._status = PreviewStatus("queued", request.quality.value, generation)

        def work() -> PreviewFrame:
            return self.render_sync(
                request,
                shot_transform=shot_transform,
                transform_config=transform_config,
                cancellation=token,
            )

        future = self._executor.submit(work)
        with self._state_lock:
            self._active_future = future

        def complete(done: Future[PreviewFrame]) -> None:
            try:
                result = done.result()
            except BaseException as exc:
                with self._state_lock:
                    current = generation == self._generation
                    if current and not isinstance(exc, PreviewCancelled):
                        self._status = PreviewStatus("error", str(exc), generation)
                if current and on_error is not None and not isinstance(exc, PreviewCancelled):
                    on_error(exc)
                return
            with self._state_lock:
                current = generation == self._generation and not token.cancelled
                if current:
                    self._status = PreviewStatus("ready", request.quality.value, generation)
            if current and on_ready is not None:
                on_ready(result)

        future.add_done_callback(complete)
        return generation

    def cancel_current(self) -> None:
        with self._state_lock:
            self._generation += 1
            if self._active_token is not None:
                self._active_token.cancel()
            if self._active_future is not None:
                self._active_future.cancel()
            self._status = PreviewStatus("cancelled", "preview request cancelled", self._generation)

    def close(self) -> None:
        self.cancel_current()
        self.close_sessions()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.backend.close()
        with self._state_lock:
            self._status = PreviewStatus("closed")
