"""Structured return type for pipeline application."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineResult:
    """Structured return value from ``apply_pipeline`` (see ``pipeline.orchestration.engine``).

    Attributes
    ----------
    data
        Transformed feature matrix (DataFrame).
    metadata
        Step artifacts (e.g. ``{"sample_weight": ...}``) or ``None`` when empty.
    sample_weight
        Per-row weights from instance reweighting, if present; otherwise ``None``.
    transformers_applied
        Names of sklearn pipeline steps, in order.
    """

    data: pd.DataFrame
    metadata: Optional[Dict[str, Any]]
    sample_weight: Optional[np.ndarray]
    transformers_applied: tuple[str, ...]

    def __iter__(self) -> Iterator:
        warnings.warn(
            "Unpacking apply_pipeline(...) as (df, meta) is deprecated; use "
            ".data, .metadata, .sample_weight, and .transformers_applied. "
            "Tuple unpacking will be removed in a future major version.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield self.data
        yield self.metadata
