"""
Pipelines subpackage.

Keep imports lightweight: pipelines can have heavy optional deps (e.g., vision + diffusers).
We expose subpackages via lazy imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any
from . import a2d, bert, dream, editflow, fastdllm, llada, llada2

__all__ = [
    # pipelines (subpackages)
    "a2d",
    "bert",
    "dream",
    "editflow",
    "fastdllm",
    "llada",
    "llada2",
    "oneflow",
    "oneflow_text_only",
    "oneflow_image_only",
    "oneflow_mixed_generation",
    "oneflow_interleaved",
    # utils module in this package
    "ctmc_utils",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in __all__:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
