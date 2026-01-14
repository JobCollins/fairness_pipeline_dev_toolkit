from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InstanceReweighting(BaseEstimator, TransformerMixin):
    """
    Compute sample weights to align observed group proportions with provided benchmarks.
    - If multiple sensitive attributes, weights are multiplied (capped).
    - If no benchmarks given, use inverse-frequency balancing per attribute.
    Outputs:
      - transform(X) returns X unchanged
      - stores `sample_weight_` aligned to input rows
    """

    def __init__(
        self,
        sensitive: list[str],
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
        max_weight: float = 10.0,
        min_count: int = 1,
    ):
        self.sensitive = sensitive
        self.benchmarks = benchmarks or {}
        self.max_weight = float(max_weight)
        self.min_count = int(min_count)
        self.sample_weight_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "InstanceReweighting":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InstanceReweighting expects a pandas DataFrame as X.")

        n = len(X)
        w = np.ones(n, dtype=float)

        for attr in self.sensitive:
            if attr not in X.columns:
                continue
            s = X[attr].astype("category")
            counts = s.value_counts()
            total = counts.sum()

            # target proportions
            bmk = self.benchmarks.get(attr)
            if bmk:
                # expected count per group
                exp = {
                    k: max(self.min_count, int(round((bmk.get(str(k), 0.0) * total))))
                    for k in set(counts.index) | set(bmk.keys())
                }
            else:
                # inverse-frequency balance
                k = len(counts)
                if k == 0:
                    continue
                target = total / k
                exp = {str(g): max(self.min_count, int(round(target))) for g in counts.index}

            # per group factor = expected / observed (clip)
            factors = {}
            for g, c in counts.items():
                gk = str(g)
                if c <= 0:
                    factors[gk] = 1.0
                else:
                    factors[gk] = float(
                        np.clip(exp.get(gk, c) / c, 1.0 / self.max_weight, self.max_weight)
                    )

            # apply multiplicatively
            w *= s.astype(str).map(factors).to_numpy()

        # normalize weights (mean=1) for stability
        if w.mean() > 0:
            w = w / w.mean()

        self.sample_weight_ = w
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sample_weight_ is None or len(self.sample_weight_) != len(X):
            # Not fatal in sklearn patterns, but warn by raising to keep behavior explicit
            raise RuntimeError(
                "InstanceReweighting must be fitted before transform; sizes must match."
            )
        return X
