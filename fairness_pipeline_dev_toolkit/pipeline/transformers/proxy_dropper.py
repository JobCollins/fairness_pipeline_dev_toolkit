from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
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
    if mask.sum() < 3:
        return 0.0
    try:
        r, _ = pearsonr(x_[mask].to_numpy(), y_[mask].to_numpy())
        if np.isnan(r):
            return 0.0
        return float(abs(r))
    except Exception:
        return 0.0


class ProxyDropper(BaseEstimator, TransformerMixin):
    """
    Drop feature columns that are too strongly associated with sensitive attributes.
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

    def _is_cat(self, s: pd.Series) -> bool:
        if (
            pd.api.types.is_categorical_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            return True
        return pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 20

    def _assoc(self, feat: pd.Series, sens: pd.Series) -> float:
        feat_cat = self._is_cat(feat)
        sens_cat = self._is_cat(sens)

        if feat_cat and sens_cat:
            return _cramers_v(feat.astype(str), sens.astype(str))

        if not feat_cat and not sens_cat:
            return _pearson_abs(feat, sens)

        if feat_cat and not sens_cat:
            _, inv = np.unique(feat.astype(str), return_inverse=True)
            return _pearson_abs(pd.Series(inv, index=feat.index), sens)

        if not feat_cat and sens_cat:
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
                a = self._assoc(X[col], X[s])
                if a > max_assoc:
                    max_assoc = a
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