from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MetricResult
from .eod_undefined import equalized_odds_point_estimate
from .input_validation import (
    nonfinite_drop_caveat,
    prepare_binary_classifier_inputs,
    prepare_regression_metric_inputs,
)


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
        # force to 1-D
        s = np.asarray(s)
        if s.ndim > 1:
            s = s.ravel()
        s = pd.Series(s)
        counts = s.value_counts()
        valid = s.map(counts) >= min_group_size
        return s, valid.to_numpy()

    def demographic_parity_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        prepared = prepare_binary_classifier_inputs(
            y_pred=y_pred, sensitive=sensitive, y_true=y_true, require_y_true=False
        )
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)
        s, valid = self._group_mask(prepared.sensitive, min_group_size)
        yp = prepared.y_pred
        if valid.sum() == 0:
            return MetricResult(
                "demographic_parity_difference",
                np.nan,
                n_per_group={},
                caveat=drop_caveat,
                n_dropped_nonfinite=prepared.n_dropped_nonfinite,
            )

        s = s[valid].to_numpy()
        yp = yp[valid]
        groups = np.unique(s)
        rates, n_per = {}, {}
        for g in groups:
            m = s == g
            n = int(m.sum())
            if n >= min_group_size:
                rates[str(g)] = float(yp[m].mean())  # selection rate (positive=1)
                n_per[str(g)] = n
        if len(rates) < 2:
            return MetricResult(
                "demographic_parity_difference",
                np.nan,
                n_per_group=n_per,
                caveat=drop_caveat,
                n_dropped_nonfinite=prepared.n_dropped_nonfinite,
            )
        diff = max(rates.values()) - min(rates.values())
        return MetricResult(
            "demographic_parity_difference",
            float(diff),
            n_per_group=n_per,
            caveat=drop_caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )

    def equalized_odds_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        prepared = prepare_binary_classifier_inputs(
            y_pred=y_pred, sensitive=sensitive, y_true=y_true, require_y_true=True
        )
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)
        s, valid = self._group_mask(prepared.sensitive, min_group_size)
        yt = prepared.y_true
        yp = prepared.y_pred
        if valid.sum() == 0:
            return MetricResult(
                "equalized_odds_difference",
                np.nan,
                n_per_group={},
                caveat=drop_caveat,
                n_dropped_nonfinite=prepared.n_dropped_nonfinite,
            )

        s = s[valid].to_numpy()
        yt = yt[valid]
        yp = yp[valid]
        groups = list(np.unique(s))
        n_per = {str(g): int((s == g).sum()) for g in groups}
        value = equalized_odds_point_estimate(yt, yp, s, groups=groups)
        return MetricResult(
            "equalized_odds_difference",
            value,
            n_per_group=n_per,
            caveat=drop_caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )

    def mae_parity_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        prepared = prepare_regression_metric_inputs(
            y_true=y_true, y_pred=y_pred, sensitive=sensitive
        )
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)
        s, valid = self._group_mask(prepared.sensitive, min_group_size)
        yt = prepared.y_true
        yp = prepared.y_pred
        if valid.sum() == 0:
            return MetricResult(
                "mae_parity_difference",
                np.nan,
                n_per_group={},
                caveat=drop_caveat,
                n_dropped_nonfinite=prepared.n_dropped_nonfinite,
            )

        s = s[valid].to_numpy()
        yt = yt[valid]
        yp = yp[valid]
        groups = np.unique(s)
        maes, n_per = {}, {}
        for g in groups:
            m = s == g
            yt_g, yp_g = yt[m], yp[m]
            maes[str(g)] = float(np.mean(np.abs(yt_g - yp_g)))
            n_per[str(g)] = int(m.sum())

        if len(maes) < 2:
            return MetricResult(
                "mae_parity_difference",
                np.nan,
                n_per_group=n_per,
                caveat=drop_caveat,
                n_dropped_nonfinite=prepared.n_dropped_nonfinite,
            )
        diff = max(maes.values()) - min(maes.values())
        return MetricResult(
            "mae_parity_difference",
            float(diff),
            n_per_group=n_per,
            caveat=drop_caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )
