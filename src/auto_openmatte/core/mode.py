"""Processing mode definitions shared by GUI and backend."""

from __future__ import annotations

from enum import Enum


class ProcessingMode(str, Enum):
    """How a synchronized, fitted shot is applied to the output."""

    EXTEND = "EXTEND"
    CONVERT = "CONVERT"

    @classmethod
    def coerce(cls, value: "ProcessingMode | str | None") -> "ProcessingMode":
        """Normalize persisted/configuration values to a processing mode."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.EXTEND
        normalized = str(value).strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported processing mode {value!r}; expected EXTEND or CONVERT"
            ) from exc
