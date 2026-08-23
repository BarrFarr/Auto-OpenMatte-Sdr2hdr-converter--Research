"""Reference-based full-frame SDR-to-HDR conversion boundary.

This module deliberately contains no new color science.  It adapts the
existing production ``composite_convert_hdr`` implementation behind a small
replaceable backend contract so the renderer and GUI do not know whether the
transform runs on CPU, GPU, or another future implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import TransformConfig
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.pipeline.compose import composite_convert_hdr
from auto_openmatte.processing.luminance import build_curve_lut


@dataclass(frozen=True)
class ConvertOutputConfig:
    """Parameters needed to apply one converted frame."""

    transform: TransformConfig = field(default_factory=TransformConfig)


class ConvertBackend(Protocol):
    """Replaceable frame-level CONVERT implementation."""

    @property
    def name(self) -> str:
        ...

    def apply(
        self,
        frame: NDArray[np.floating],
        shot_transform: ShotTransform,
        output_config: ConvertOutputConfig,
    ) -> NDArray[np.floating]:
        """Apply a full-frame conversion and return HDR signal-domain pixels."""
        ...


class CPUConvertBackend:
    """CPU adapter for the existing reference transform/composite path."""

    name = "cpu-reference"

    def __init__(self) -> None:
        self._luts: dict[str, dict] = {}

    @staticmethod
    def _curve_key(shot_transform: ShotTransform) -> str:
        return repr(shot_transform.luminance_curve)

    def apply(
        self,
        frame: NDArray[np.floating],
        shot_transform: ShotTransform,
        output_config: ConvertOutputConfig,
    ) -> NDArray[np.floating]:
        transform_config = output_config.transform
        transform_config.validate()

        lut = None
        if shot_transform.luminance_curve and len(shot_transform.luminance_curve) >= 2:
            key = self._curve_key(shot_transform)
            lut = self._luts.get(key)
            if lut is None:
                lut = build_curve_lut(shot_transform.luminance_curve)
                self._luts[key] = lut

        # This is the existing full-frame path.  HDR is used by fitting only;
        # no reference pixels are passed to or composited by this method.
        return composite_convert_hdr(
            frame,
            shot_transform,
            sdr_transfer=transform_config.sdr_transfer,
            hdr_transfer=transform_config.hdr_transfer,
            peak_nits=transform_config.peak_nits,
            prebuilt_lut=lut,
        )


class ConvertRenderer:
    """Stable renderer-facing facade around a replaceable CONVERT backend."""

    def __init__(self, backend: ConvertBackend | None = None) -> None:
        self.backend = backend or CPUConvertBackend()

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def apply(
        self,
        frame: NDArray[np.floating],
        shot_transform: ShotTransform,
        output_config: ConvertOutputConfig,
    ) -> NDArray[np.floating]:
        return self.backend.apply(frame, shot_transform, output_config)
