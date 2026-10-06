from __future__ import annotations

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.exceptions import DependencyError

from .base import MetricResult
from .input_validation import (
    nonfinite_drop_caveat,
    prepare_binary_classifier_inputs,
    prepare_regression_metric_inputs,
)


class FairlearnAdapter:
    """
    Adapter over Fairlearn metrics.

    - ``demographic_parity_difference`` delegates to
      ``fairlearn.metrics.demographic_parity_difference`` (same binary gap definition).
    - ``equalized_odds_difference`` is computed natively under fairpipe's incomplete-EO
      rule (NaN when any analysed group lacks positives or negatives). Fairlearn's
      library function returns a finite gap in that case, which would look "perfectly
      fair"; the backend name is kept for compatibility.
    - ``mae_parity_difference`` is computed natively (Fairlearn has no MAE parity gap).
    """

    name = "fairlearn"

    def __init__(self):
        try:
            import fairlearn.metrics as flm

            self._flm = flm
            self._ok = True
        except Exception:
            self._flm = None
            self._ok = False

    def available(self) -> bool:
        return bool(self._ok)

    def _require_available(self) -> None:
        if self.available():
            return
        raise DependencyError(
            "Fairlearn adapter requires fairlearn.",
            dependency_name="fairlearn",
            extra_name="adapters",
        )

    def _mask_small_groups(self, sensitive, min_group_size: int):
        s = pd.Series(sensitive)
        counts = s.value_counts()
        valid = s.map(counts) >= min_group_size
        return s, valid.to_numpy()

    def demographic_parity_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        self._require_available()
        prepared = prepare_binary_classifier_inputs(
            y_pred=y_pred, sensitive=sensitive, y_true=y_true, require_y_true=False
        )
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)
        s, valid = self._mask_small_groups(prepared.sensitive, min_group_size)
        yp = prepared.y_pred
        # Fairlearn's DPD signature requires y_true; pass a dummy when absent.
        yt = prepared.y_true if prepared.y_true is not None else np.zeros(len(yp), dtype=float)
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
        yt = yt[valid]
        groups = np.unique(s)
        n_per = {}
        keep = np.zeros(len(s), dtype=bool)
        for g in groups:
            m = s == g
            n = int(m.sum())
            if n >= min_group_size:
                n_per[str(g)] = n
                keep |= m

        if len(n_per) < 2:
            return MetricResult(
                "demographic_parity_difference",
                np.nan,
                n_per_group=n_per,
                caveat=drop_caveat,
                n_dropped_nonfinite=prepared.n_dropped_nonfinite,
            )

        # Delegate to Fairlearn's library function on the filtered rows.
        value = float(
            self._flm.demographic_parity_difference(yt[keep], yp[keep], sensitive_features=s[keep])
        )
        return MetricResult(
            "demographic_parity_difference",
            value,
            n_per_group=n_per,
            caveat=drop_caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )

    def equalized_odds_difference(
        self, y_true, y_pred, sensitive, *, min_group_size: int = 30
    ) -> MetricResult:
        self._require_available()
        prepared = prepare_binary_classifier_inputs(
            y_pred=y_pred, sensitive=sensitive, y_true=y_true, require_y_true=True
        )
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)
        s, valid = self._mask_small_groups(prepared.sensitive, min_group_size)
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
        # Native incomplete-EO rule (Fairlearn's library EOD stays finite here).
        from .eod_undefined import equalized_odds_point_estimate

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
        self._require_available()
        prepared = prepare_regression_metric_inputs(
            y_true=y_true, y_pred=y_pred, sensitive=sensitive
        )
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)
        s, valid = self._mask_small_groups(prepared.sensitive, min_group_size)
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
