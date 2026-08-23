"""Bounded source/rendered preview caches."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar

from auto_openmatte.preview.models import PreviewCacheKey, PreviewFrame


T = TypeVar("T")


class BoundedPreviewCache(Generic[T]):
    """Thread-safe bounded cache with explicit invalidation."""

    def __init__(self, max_entries: int):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._items: OrderedDict[PreviewCacheKey, T] = OrderedDict()
        self._lock = RLock()

    def get(self, key: PreviewCacheKey) -> T | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: PreviewCacheKey, value: T) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class PreviewCaches:
    """Separate source and corrected/rendered caches."""

    def __init__(self, source_entries: int = 6, rendered_entries: int = 6):
        self.source = BoundedPreviewCache[PreviewFrame](source_entries)
        self.rendered = BoundedPreviewCache[PreviewFrame](rendered_entries)

    def invalidate_source(self) -> None:
        self.source.invalidate()

    def invalidate_rendered(self) -> None:
        self.rendered.invalidate()

    def invalidate_all(self) -> None:
        self.invalidate_source()
        self.invalidate_rendered()
