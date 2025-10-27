from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from .base import MetricAdapter, MetricResult

class NativeAdapter:
    """
    Built-in baseline adapter.
    - Requires no external libraries.
    - Ensures FairnessAnalyzer always works even if Fairlearn/Aequitas aren't installed.
    """
    name = "native"

    def available(self) -> bool:
        return True  # Always available

    def _group_mask(self, s, min_group_size):
        s = pd.Series(s)
        counts = s.value_counts()
        valid = s.map(counts) >= min_group_size
        return s, valid.to_numpy()

    def demographic_parity_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        s, valid = self._group_mask(sensitive, min_group_size)
        yp = np.asarray(y_pred)
        if valid.sum() == 0:
            return MetricResult("demographic_parity_difference", np.nan, n_per_group={})

        s = s[valid].to_numpy()
        yp = yp[valid]
        groups = np.unique(s)
        rates, n_per = {}, {}
        for g in groups:
            m = (s == g)
            n = int(m.sum())
            if n >= min_group_size:
                rates[str(g)] = float(yp[m].mean())  # selection rate
                n_per[str(g)] = n
        if len(rates) < 2:
            return MetricResult("demographic_parity_difference", np.nan, n_per_group=n_per)
        diff = max(rates.values()) - min(rates.values())
        return MetricResult("demographic_parity_difference", float(diff), n_per_group=n_per)

    def equalized_odds_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        s, valid = self._group_mask(sensitive, min_group_size)
        yt = np.asarray(y_true); yp = np.asarray(y_pred)
        if valid.sum() == 0:
            return MetricResult("equalized_odds_difference", np.nan, n_per_group={})

        s = s[valid].to_numpy(); yt = yt[valid]; yp = yp[valid]
        groups = np.unique(s)
        tpr, fpr, n_per = {}, {}, {}
        for g in groups:
            m = (s == g)
            yt_g, yp_g = yt[m], yp[m]
            pos = (yt_g == 1)
            neg = (yt_g == 0)
            tpr[str(g)] = float(np.mean(yp_g[pos]) if pos.any() else np.nan)
            fpr[str(g)] = float(np.mean(yp_g[neg]) if neg.any() else np.nan)
            n_per[str(g)] = int(m.sum())

        def span(d):
            vals = [v for v in d.values() if not np.isnan(v)]
            return np.nan if len(vals) < 2 else (max(vals) - min(vals))
        tpr_gap = span(tpr)
        fpr_gap = span(fpr)
        value = np.nan if (np.isnan(tpr_gap) or np.isnan(fpr_gap)) else max(tpr_gap, fpr_gap)
        return MetricResult("equalized_odds_difference", float(value) if value==value else np.nan, n_per_group=n_per)
