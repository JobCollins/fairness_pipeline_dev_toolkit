from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample
from statsmodels.stats.power import TTestIndPower


@dataclass
class ABPowerSpec:
    effect_size: float = 0.10
    alpha: float = 0.05
    min_group_size: int = 30


class FairnessABTestAnalyzer:
    """
    A/B test utilities: intersectional power estimation and heterogeneous effects.
    """

    def __init__(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
        protected_attributes: Sequence[str],
        outcome_column: str,
        fairness_metric_column: str,  # e.g., already-computed dp/eo per row or group outcome for parity proxy
        business_metrics: Sequence[str] = (),
    ):
        self.control = control.copy()
        self.treatment = treatment.copy()
        self.protected = list(protected_attributes)
        self.outcome_col = outcome_column
        self.fair_col = fairness_metric_column
        self.business = list(business_metrics)

    def _intersections(self) -> List[Tuple[Any, ...]]:
        if not self.protected:
            return [("__overall__",)]
        combos = pd.concat(
            [self.control[self.protected], self.treatment[self.protected]], axis=0
        ).drop_duplicates()
        return [tuple(row) for _, row in combos.iterrows()]

    def _slice(self, df: pd.DataFrame, inter: Tuple[Any, ...]) -> pd.DataFrame:
        if inter == ("__overall__",):
            return df
        m = pd.Series([True] * len(df), index=df.index)
        for col, val in zip(self.protected, inter):
            m &= df[col] == val
        return df[m]

    def power_by_intersection(self, spec: ABPowerSpec) -> Dict[str, float]:
        power = TTestIndPower()
        out: Dict[str, float] = {}
        for inter in self._intersections():
            c = self._slice(self.control, inter)
            t = self._slice(self.treatment, inter)
            n1, n2 = len(c), len(t)
            key = "×".join(map(str, inter))
            if n1 < spec.min_group_size or n2 < spec.min_group_size:
                out[key] = np.nan
                continue
            ratio = n2 / n1 if n1 else 0.0
            pwr = power.solve_power(
                effect_size=spec.effect_size,
                nobs1=n1,
                ratio=ratio,
                alpha=spec.alpha,
                alternative="two-sided",
            )
            out[key] = float(pwr)
        return out

    def heterogeneous_effects(self, n_bootstrap: int = 1000, alpha: float = 0.05) -> pd.DataFrame:
        rows = []
        lo = (alpha / 2) * 100
        hi = 100 - lo
        for inter in self._intersections():
            c = self._slice(self.control, inter)
            t = self._slice(self.treatment, inter)
            if c.empty or t.empty:
                continue

            # business metric effects (means)
            for mcol in [self.outcome_col, *self.business, self.fair_col]:
                if mcol not in c.columns or mcol not in t.columns:
                    continue
                effs = []
                for _ in range(n_bootstrap):
                    bc = resample(c[mcol], replace=True)
                    bt = resample(t[mcol], replace=True)
                    effs.append(bt.mean() - bc.mean())
                rows.append(
                    {
                        "intersection": "×".join(map(str, inter)),
                        "metric": mcol,
                        "effect": float(np.mean(effs)),
                        "lower": float(np.percentile(effs, lo)),
                        "upper": float(np.percentile(effs, hi)),
                        "significant": bool(
                            not (np.percentile(effs, lo) <= 0 <= np.percentile(effs, hi))
                        ),
                        "n_control": int(len(c)),
                        "n_treatment": int(len(t)),
                    }
                )
        return pd.DataFrame(rows)
