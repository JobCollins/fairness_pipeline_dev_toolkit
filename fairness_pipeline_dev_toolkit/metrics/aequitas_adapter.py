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


class AequitasAdapter:
    """
    Adapter named for Aequitas.

    Aequitas does not expose demographic-parity / equalized-odds *difference* APIs
    matching fairpipe's MetricResult contract, so all three metrics are **computed
    natively**; the backend name is kept for compatibility. Availability still
    requires ``import aequitas`` (``pip install fairpipe[adapters]``).
    """

    name = "aequitas"

    def __init__(self):
        try:
            # aequitas imports would go here, but we avoid hard dependency at Phase 1
            import aequitas  # noqa: F401

            self._ok = True
        except Exception:
            self._ok = False

    def available(self) -> bool:
        return bool(self._ok)

    def _require_available(self) -> None:
        if self.available():
            return
        raise DependencyError(
            "Aequitas adapter requires aequitas.",
            dependency_name="aequitas",
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
                rates[str(g)] = float(np.mean(yp[m]))  # selection rate
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
