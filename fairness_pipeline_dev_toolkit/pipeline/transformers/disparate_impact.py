from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class DisparateImpactRemover(BaseEstimator, TransformerMixin):
    """
    Scafold stub (no-op): API for Feldman et al. style rank-preserving repair.
    Later implement quantile-mapping by group with repair_level in [0,1].
    """

    def __init__(self, columns=None, sensitive_col=None, repair_level: float = 1.0):
        self.columns = columns
        self.sensitive_col = sensitive_col
        self.repair_level = repair_level

    def fit(self, X, y=None):
        # Phase 0: record nothing
        return self

    def transform(self, X, y=None):
        return X
