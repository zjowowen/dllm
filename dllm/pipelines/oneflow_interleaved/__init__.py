"""
OneFlow interleaved-generation pipeline.

Full Algorithm 3 training (τ_text ~ Unif[0,2]) and Algorithm 1-2 sampling
with coordinated text insertion + image flow matching via the interleaved
schedule.

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
    "OneFlowInterleavedTrainer",
    "OneFlowInterleavedSampler",
    "OneFlowInterleavedSamplerConfig",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"trainer", "sampler", "runtime_config"}:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name == "OneFlowInterleavedTrainer":
        from .trainer import OneFlowInterleavedTrainer as _T

        globals()[name] = _T
        return _T
    if name == "OneFlowInterleavedSampler":
        from .sampler import OneFlowInterleavedSampler as _S

        globals()[name] = _S
        return _S
    if name == "OneFlowInterleavedSamplerConfig":
        from .sampler import OneFlowInterleavedSamplerConfig as _C

        globals()[name] = _C
        return _C
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
