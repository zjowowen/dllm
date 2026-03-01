"""
OneFlow image-only pipeline.

Isolates image flow matching training: text is fully preserved (τ_text > 1),
only the image velocity head is meaningfully optimized.

This package is intentionally lightweight at import time. Heavy modules
(trainer/model/sampler backbones) are loaded lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # submodules
    "trainer",
    "sampler",
    "runtime_config",
    # main entrypoints (lazy)
    "OneFlowImageOnlyTrainer",
    "OneFlowImageOnlySampler",
    "OneFlowImageOnlySamplerConfig",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"trainer", "sampler", "runtime_config"}:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name == "OneFlowImageOnlyTrainer":
        from .trainer import OneFlowImageOnlyTrainer as _T

        globals()[name] = _T
        return _T
    if name == "OneFlowImageOnlySampler":
        from .sampler import OneFlowImageOnlySampler as _S

        globals()[name] = _S
        return _S
    if name == "OneFlowImageOnlySamplerConfig":
        from .sampler import OneFlowImageOnlySamplerConfig as _C

        globals()[name] = _C
        return _C
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
