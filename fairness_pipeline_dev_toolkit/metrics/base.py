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

    CI contract: ``ci`` is either a finite ``(lower, upper)`` pair or ``None``. It is
    never ``(nan, nan)``. When a CI was requested but is undefined, ``ci`` is ``None``
    and ``ci_note`` starts with ``"undefined:<reason>"``. ``ci_kind`` names how ``ci``
    was built (``"simultaneous_pairwise"``, ``"template_bonferroni_t"``,
    ``"percentile"``, ``"bca"``) and is ``None`` when there is no CI.
    """

    metric: str  # e.g., "demographic_parity_difference"
    value: float  # point estimate
    ci: Optional[tuple[float, float]] = None  # confidence interval (Phase 3 fills this)
    effect_size: Optional[float] = None  # risk ratio, Cohen's d, etc. (Phase 3 fills this)
    n_per_group: Optional[Dict[str, int]] = None  # sample sizes by group
    caveat: Optional[str] = None  # provenance warning; None for ordinary user data
    n_dropped_nonfinite: Optional[int] = None  # rows dropped for NaN/inf in y_true/y_pred
    p_value: Optional[float] = None  # permutation-test p-value for "all groups equal"
    ci_kind: Optional[str] = None  # how ``ci`` was built; None when there is no CI
    ci_note: Optional[str] = None  # why the CI is undefined, or a small-sample note


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

    def mae_parity_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | pd.Series,
        *,
        min_group_size: int = 30,
    ) -> MetricResult: ...
