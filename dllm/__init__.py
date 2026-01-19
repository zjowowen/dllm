"""
`dllm` top-level package.

Keep imports *lightweight* by default.

Why:
- Unit tests and small utilities should be able to import `dllm.*` submodules without
  eagerly importing heavy dependencies (e.g., Transformers model classes).
- Optional backends (e.g., NPU) may not be available in every environment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["core", "data", "pipelines", "utils"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in __all__:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
