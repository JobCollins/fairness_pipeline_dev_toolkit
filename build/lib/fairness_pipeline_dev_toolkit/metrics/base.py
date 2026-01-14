from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# -------------------------------
# Result schema (Part 5 requirement)
# -------------------------------
@dataclass
class MetricResult:
    """
    Unified result object returned by all adapters and by FairnessAnalyzer.
    This keeps outputs stable across libraries and easy to log into MLflow.
    """

    metric: str  # e.g., "demographic_parity_difference"
    value: float  # point estimate
    ci: Optional[tuple[float, float]] = None  # confidence interval (Phase 3 fills this)
    effect_size: Optional[float] = None  # risk ratio, Cohen's d, etc. (Phase 3 fills this)
    n_per_group: Optional[Dict[str, int]] = None  # sample sizes by group


# ---------------------------------------
# Minimal adapter interface for libraries
# ---------------------------------------
@runtime_checkable
class MetricAdapter(Protocol):
    """All adapters must implement these methods."""

    name: str

    def available(self) -> bool:
        """Return True if the underlying library is importable and usable."""
        ...

    def demographic_parity_difference(
        self,
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        sensitive: np.ndarray | pd.Series,
        *,
        min_group_size: int = 30,
    ) -> MetricResult: ...

    def equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | pd.Series,
        *,
        min_group_size: int = 30,
    ) -> MetricResult: ...
