"""
OneFlow text-only pipeline.

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
    "visualize",
    "models",
    # main entrypoints (lazy)
    "OneFlowTextOnlyTrainer",
    "OneFlowTextOnlySampler",
    "OneFlowTextOnlySamplerConfig",
    "OneFlowTextOnlyModel",
    "OneFlowTextOnlyConfig",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"trainer", "sampler", "runtime_config", "visualize", "models"}:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name == "OneFlowTextOnlyTrainer":
        from .trainer import OneFlowTextOnlyTrainer as _T

        globals()[name] = _T
        return _T
    if name == "OneFlowTextOnlySampler":
        from .sampler import OneFlowTextOnlySampler as _S

        globals()[name] = _S
        return _S
    if name == "OneFlowTextOnlySamplerConfig":
        from .sampler import OneFlowTextOnlySamplerConfig as _C

        globals()[name] = _C
        return _C
    if name == "OneFlowTextOnlyModel":
        from .models import OneFlowTextOnlyModel as _M

        globals()[name] = _M
        return _M
    if name == "OneFlowTextOnlyConfig":
        from .models import OneFlowTextOnlyConfig as _MC

        globals()[name] = _MC
        return _MC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)

