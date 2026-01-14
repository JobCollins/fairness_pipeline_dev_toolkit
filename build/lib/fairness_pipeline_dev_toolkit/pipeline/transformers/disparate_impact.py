from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DisparateImpactRemover(BaseEstimator, TransformerMixin):
    """
    Quantile-based repair for continuous features to reduce group-level distribution gaps.
    - Works per feature in `features`
    - Uses `sensitive` (single attribute for Phase 2; extendable later)
    - For each group g:
        1) compute within-group ranks (uniform [0,1] via (rank-0.5)/n_g)
        2) map these quantiles into the POOL (all groups) empirical quantile
        3) blend: repaired = (1-repair_level)*orig + repair_level*mapped
    """

    def __init__(
        self,
        sensitive: str,
        features: List[str],
        repair_level: float = 1.0,
        min_group_size: int = 20,
    ):
        self.sensitive = sensitive
        self.features = list(features)
        self.repair_level = float(repair_level)
        self.min_group_size = int(min_group_size)

        # fitted state
        self.groups_: Optional[pd.Index] = None
        self.pool_quantiles_: Dict[str, np.ndarray] = {}
        self.pool_values_: Dict[str, np.ndarray] = {}

    def _empirical_quantiles(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (q, v) such that q in [0,1] (monotone) and v are corresponding sorted values.
        """
        x_sorted = np.sort(x.astype(float))
        n = len(x_sorted)
        if n == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 0.0])
        q = (np.arange(1, n + 1) - 0.5) / n
        return q, x_sorted

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "DisparateImpactRemover":
        if self.sensitive not in X.columns:
            raise ValueError(f"Sensitive attribute '{self.sensitive}' not in DataFrame.")
        for f in self.features:
            if f not in X.columns:
                raise ValueError(f"Feature '{f}' not in DataFrame.")

        self.groups_ = X[self.sensitive].astype("category").cat.categories

        # compute pooled quantiles per feature
        for f in self.features:
            q, v = self._empirical_quantiles(X[f].dropna().to_numpy())
            self.pool_quantiles_[f] = q
            self.pool_values_[f] = v
        return self

    def _interp(self, q_grid: np.ndarray, v_grid: np.ndarray, q: np.ndarray) -> np.ndarray:
        # piecewise linear interpolation of quantiles into values
        if len(q_grid) == 0:
            return np.full_like(q, np.nan, dtype=float)
        return np.interp(q, q_grid, v_grid, left=v_grid[0], right=v_grid[-1])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.groups_ is None:
            raise RuntimeError("DisparateImpactRemover must be fitted before transform.")

        df = X.copy()
        g = df[self.sensitive].astype("category")

        for f in self.features:
            pool_q = self.pool_quantiles_[f]
            pool_v = self.pool_values_[f]
            x = df[f].to_numpy(dtype=float, copy=True)

            # operate per group
            for cat in g.cat.categories:
                idx = np.where(g.to_numpy() == cat)[0]
                if idx.size < self.min_group_size:
                    continue  # avoid unstable mapping

                vals = x[idx]
                # group quantiles in [0,1]
                ranks = pd.Series(vals).rank(method="average").to_numpy()
                q = (ranks - 0.5) / max(1, len(vals))

                mapped = self._interp(pool_q, pool_v, q)
                repaired = (1.0 - self.repair_level) * vals + self.repair_level * mapped
                x[idx] = repaired

            df[f] = x

        return df
