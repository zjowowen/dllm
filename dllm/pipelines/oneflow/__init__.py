"""
OneFlow pipeline (multimodal): insertion-based text edit flow + image-latent flow matching.

IMPORTANT: keep imports lightweight.

We intentionally avoid importing trainer/model backbones at import time, because they
can pull in heavy optional deps (Transformers / Transfusion / flash-attn / NPU glue).
Use lazy attribute access instead.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # submodules
    "trainer",
    "sampler",
    "utils",
    "sequence_ops",
    "losses",
    "trace",
    "runtime_config",
    "models",
    # main entrypoints (lazy)
    "OneFlowTrainer",
    "OneFlowSampler",
    "OneFlowSamplerConfig",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"trainer", "sampler", "utils", "sequence_ops", "losses", "trace", "runtime_config", "models"}:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name == "OneFlowTrainer":
        from .trainer import OneFlowTrainer as _T

        globals()[name] = _T
        return _T
    if name == "OneFlowSampler":
        from .sampler import OneFlowSampler as _S

        globals()[name] = _S
        return _S
    if name == "OneFlowSamplerConfig":
        from .sampler import OneFlowSamplerConfig as _C

        globals()[name] = _C
        return _C
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)


