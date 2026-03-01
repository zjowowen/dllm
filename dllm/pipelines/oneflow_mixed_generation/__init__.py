"""
OneFlow mixed-generation pipeline.

Validates that text insertion loss and image flow matching loss can be
jointly optimized via `mixed_generation_prob` controlling the fraction
of image-bearing samples per batch.

This package is intentionally lightweight at import time.
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
    "OneFlowMixedGenerationTrainer",
    "OneFlowMixedGenerationSampler",
    "OneFlowMixedGenerationSamplerConfig",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"trainer", "sampler", "runtime_config"}:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name == "OneFlowMixedGenerationTrainer":
        from .trainer import OneFlowMixedGenerationTrainer as _T

        globals()[name] = _T
        return _T
    if name == "OneFlowMixedGenerationSampler":
        from .sampler import OneFlowMixedGenerationSampler as _S

        globals()[name] = _S
        return _S
    if name == "OneFlowMixedGenerationSamplerConfig":
        from .sampler import OneFlowMixedGenerationSamplerConfig as _C

        globals()[name] = _C
        return _C
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
