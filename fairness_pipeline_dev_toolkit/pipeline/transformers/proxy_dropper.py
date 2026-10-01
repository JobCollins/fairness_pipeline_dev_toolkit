from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin


def _is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s).dropna().unique()
    return len(vals) == 2


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association for two categoricals."""
    tbl = pd.crosstab(x, y)
    if tbl.size == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(tbl, correction=False)
    n = tbl.values.sum()
    if n == 0:
        return 0.0
    r, k = tbl.shape
    denom = min(r - 1, k - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def _pearson_abs(x: pd.Series, y: pd.Series) -> float:
    x_ = pd.to_numeric(x, errors="coerce")
    y_ = pd.to_numeric(y, errors="coerce")
    mask = (~x_.isna()) & (~y_.isna())
    if mask.sum() < 2:
        return 0.0
    
    x_vals = x_[mask].to_numpy()
    y_vals = y_[mask].to_numpy()
    
    if np.std(x_vals) == 0 or np.std(y_vals) == 0:
        return 0.0
        
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


class ProxyDropper(BaseEstimator, TransformerMixin):
    """
    Drop feature columns that are too strongly associated with sensitive attributes.

    Parameters:
      - sensitive: list of sensitive column names present in X
      - features: list of candidate feature columns to test/drop (default = all non-sensitive)
      - threshold: association threshold; columns with max association >= threshold are dropped
      - max_drop: optional cap on how many columns to drop to avoid excessive pruning

    Learned attributes:
      - dropped_columns_: list[str] of columns removed by transform()
      - assoc_scores_: dict[col] -> max association across sensitive attrs
    """

    def __init__(
        self,
        sensitive: List[str],
        features: Optional[List[str]] = None,
        threshold: float = 0.3,
        max_drop: Optional[int] = None,
    ):
        self.sensitive = list(sensitive)
        self.features = list(features) if features is not None else None
        self.threshold = float(threshold)
        self.max_drop = max_drop

        self.dropped_columns_: List[str] = []
        self.assoc_scores_: Dict[str, float] = {}

    def _assoc(self, feat: pd.Series, sens: pd.Series) -> float:
        feat_cat = not pd.api.types.is_numeric_dtype(feat)
        sens_cat = not pd.api.types.is_numeric_dtype(sens)

        if feat_cat and sens_cat:
            return _cramers_v(feat, sens)

        if not feat_cat and not sens_cat:
            return _pearson_abs(feat, sens)

        if feat_cat and not sens_cat and _is_binary_series(feat):
            _, inv = np.unique(feat.astype(str), return_inverse=True)
            return _pearson_abs(pd.Series(inv, index=feat.index), sens)

        if not feat_cat and sens_cat and _is_binary_series(sens):
            _, inv = np.unique(sens.astype(str), return_inverse=True)
            return _pearson_abs(feat, pd.Series(inv, index=sens.index))

        return _cramers_v(feat.astype(str), sens.astype(str))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ProxyDropper expects a pandas DataFrame X.")

        missing = [a for a in self.sensitive if a not in X.columns]
        if missing:
            raise ValueError(f"Sensitive attribute(s) not found in DataFrame: {missing}")

        cand_feats = self.features
        if cand_feats is None:
            cand_feats = [c for c in X.columns if c not in self.sensitive]

        scores: Dict[str, float] = {}
        for col in cand_feats:
            max_assoc = 0.0
            for s in self.sensitive:
                try:
                    a = self._assoc(X[col], X[s])
                    if a > max_assoc:
                        max_assoc = a
                except Exception:
                    continue
            scores[col] = float(max_assoc)

        to_drop = [c for c, a in scores.items() if a >= self.threshold]
        if self.max_drop is not None and len(to_drop) > self.max_drop:
            to_drop = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
            to_drop = to_drop[: self.max_drop]

        self.assoc_scores_ = scores
        self.dropped_columns_ = to_drop
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.dropped_columns_:
            return X
        keep = [c for c in X.columns if c not in self.dropped_columns_]
        return X[keep].copy()