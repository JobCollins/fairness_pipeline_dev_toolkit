from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ReweighingTransformer(BaseEstimator, TransformerMixin):
    """
    Compute per-row sample weights to counter representation imbalance.

    Behavior:
      - If `benchmarks` is provided (e.g., {"group_col": {"A": 0.5, "B": 0.5}}),
        we compute weights to match those target proportions (per sensitive column).
      - Else, we default to uniform target proportions across groups in each sensitive column.

    Output:
      - `transform()` returns X unchanged.
      - Learned attribute `sample_weight_` is a 1D numpy array aligned to X rows.

    Notes:
      - Multiple sensitive columns are supported; weights are multiplied across attributes
        (clipped to avoid explosions).
      - This is a scikit-learn compatible transformer: place it inside a Pipeline step.
    """

    def __init__(
        self,
        sensitive: List[str],
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
        clip: float = 10.0,
    ):
        self.sensitive = list(sensitive)
        self.benchmarks = benchmarks or {}
        self.clip = float(clip)

        # learned on fit
        self.sample_weight_: Optional[np.ndarray] = None
        self._per_attr_weights_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ReweighingTransformer expects a pandas DataFrame X.")

        n = len(X)
        if n == 0:
            self.sample_weight_ = np.array([], dtype=float)
            return self

        # initialize all-ones
        w = np.ones(n, dtype=float)
        self._per_attr_weights_.clear()

        for attr in self.sensitive:
            if attr not in X.columns:
                raise ValueError(f"Sensitive attribute '{attr}' not found in DataFrame.")

            counts = X[attr].value_counts(dropna=False).to_dict()
            total = float(sum(counts.values()))
            if total <= 0:
                continue

            # Targets: benchmarks (if provided) else uniform over observed groups
            observed_groups = list(counts.keys())
            if attr in self.benchmarks and isinstance(self.benchmarks[attr], dict):
                targets = dict(self.benchmarks[attr])
                # normalize benchmarks just in case
                s = sum(max(float(v), 0.0) for v in targets.values())
                if s > 0:
                    targets = {g: float(v) / s for g, v in targets.items()}
                else:
                    # fallback to uniform
                    targets = {g: 1.0 / len(observed_groups) for g in observed_groups}
            else:
                targets = {g: 1.0 / len(observed_groups) for g in observed_groups}

            # per-group weight = target_prop / observed_prop
            # (observed_prop = counts[g] / total)
            per_group_w = {}
            for g, cnt in counts.items():
                obs_prop = float(cnt) / total if total else 0.0
                tgt_prop = float(targets.get(g, 0.0))
                if obs_prop <= 0.0:
                    # if group absent (rare if we got it from counts), weight 1
                    per_group_w[g] = 1.0
                else:
                    per_group_w[g] = max(1e-9, tgt_prop / obs_prop)

            self._per_attr_weights_[attr] = per_group_w

            # multiply per-row weights; clip to avoid extremes
            gvals = X[attr].astype(object).to_numpy()
            w *= np.vectorize(lambda v: per_group_w.get(v, 1.0))(gvals)

        # final clipping
        w = np.clip(w, 1.0 / self.clip, self.clip)
        # normalize to mean 1 for stability
        mu = float(np.mean(w)) if len(w) else 1.0
        self.sample_weight_ = (w / mu) if mu > 0 else w
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sample_weight_ is None or len(self.sample_weight_) != len(X):
            # If transform is called without fit or on a different X, recompute quickly
            self.fit(X)
        return X
