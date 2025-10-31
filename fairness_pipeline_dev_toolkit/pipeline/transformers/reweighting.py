from __future__ import annotations

from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin


class InstanceReweighting(BaseEstimator, TransformerMixin):
    """
    Scafold stub (no-op). Later compute sample_weight based on config goals.
    """

    def __init__(
        self,
        strategy: str = "none",
        target_col: Optional[str] = None,
        sensitive_col: Optional[str] = None,
    ):
        self.strategy = strategy
        self.target_col = target_col
        self.sensitive_col = sensitive_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convention: return X unchanged; weights are provided via a companion method later.
        return X
