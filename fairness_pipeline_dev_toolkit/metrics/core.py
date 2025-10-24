from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ..utils.validation import coerce_arrays, check_lengths

@dataclass
class Result:
    metric: str
    value: float
    ci: Optional[tuple[float, float]] = None
    effect_size: Optional[float] = None
    n_per_group: Optional[Dict[str, int]] = None

class FairnessAnalyzer:
    """User-facing entry point (stub for Phase 0)."""

    def __init__(self, *, min_group_size: int = 30, nan_policy: str = "exclude"):
        self.min_group_size = min_group_size
        self.nan_policy = nan_policy

    def demographic_parity_difference(self, y_pred, sensitive):
        y_pred = np.asarray(y_pred)
        sensitive = np.asarray(sensitive)
        groups = np.unique(sensitive)
        rates, n_per = [], {}
        for g in groups:
            m = sensitive == g
            n = m.sum()
            if n >= self.min_group_size:
                rates.append(y_pred[m].mean())
                n_per[str(g)] = int(n)
        value = np.nan if len(rates) < 2 else float(np.max(rates) - np.min(rates))
        return Result(metric="demographic_parity_difference", value=value, n_per_group=n_per)