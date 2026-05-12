"""Compatibility shim for `fairpipe.monitoring` (lazy proxy; avoids importing optional dashboard deps)."""

from __future__ import annotations

from typing import Any, List

import fairness_pipeline_dev_toolkit.monitoring as _src

__all__ = list(_src.__all__)


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_src, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
