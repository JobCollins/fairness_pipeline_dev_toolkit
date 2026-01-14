from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..stats.bootstrap import bootstrap_ci
from ..stats.effect_size import cohens_d, risk_ratio
from ..utils.intersectional import build_intersectional_labels, min_group_mask
from .aequitas_adapter import AequitasAdapter
from .fairlearn_adapter import FairlearnAdapter
from .native_adapter import NativeAdapter


@dataclass
class Result:
    metric: str
    value: float
    ci: Optional[tuple[float, float]] = None
    effect_size: Optional[float] = None
    n_per_group: Optional[Dict[str, int]] = None


class FairnessAnalyzer:
    """
    User-facing orchestrator. Adds:
    - Intersectional grouping
    - min_group_size filtering
    - Optional bootstrap CIs (percentile/BCa)
    - Optional effect sizes (risk ratio for rates, Cohen's d for errors)
    """

    def __init__(
        self,
        *,
        min_group_size: int = 30,
        nan_policy: str = "exclude",
        backend: Optional[str] = None,
    ):
        self.min_group_size = min_group_size
        self.nan_policy = nan_policy

        self._adapters = {
            "fairlearn": FairlearnAdapter(),
            "aequitas": AequitasAdapter(),
            "native": NativeAdapter(),
        }
        if backend is None:
            for k, a in self._adapters.items():
                if hasattr(a, "available") and a.available():
                    self._backend = k
                    self._adapter = a
                    break
        else:
            if backend not in self._adapters:
                raise ValueError(f"Unknown backend: {backend}")
            a = self._adapters[backend]
            if hasattr(a, "available") and not a.available():
                raise RuntimeError(f"Requested backend '{backend}' is not available")
            self._backend = backend
            self._adapter = a

        self._cache: Dict[str, Any] = {}

    @property
    def backend(self) -> str:
        return self._backend

    # ---------- helpers ----------

    def _intersectional_prep(
        self,
        attrs_df: pd.DataFrame,
        columns: Optional[List[str]],
    ) -> np.ndarray:
        labels = build_intersectional_labels(
            attrs_df, columns=columns, include_na=(self.nan_policy != "exclude")
        )
        # Convert categorical Series to numpy array, handling NaN values properly
        # If labels is categorical, convert to string first to avoid indexing issues
        if pd.api.types.is_categorical_dtype(labels):
            labels = labels.astype(str)
        # Convert to numpy array, replacing NaN strings with actual NaN
        labels_array = np.asarray(labels, dtype=object)
        # Replace 'nan' strings (from categorical conversion) with actual NaN
        labels_array = np.where(labels_array == "nan", np.nan, labels_array)
        return labels_array

    # ---------- DPD ----------

    def demographic_parity_difference(
        self,
        y_pred,
        sensitive,
        *,
        intersectional: bool = False,
        attrs_df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        with_ci: bool = True,
        ci_level: float = 0.95,
        ci_method: str = "percentile",
        ci_samples: int = 1000,
        with_effect_size: bool = True,
    ):
        yp = np.asarray(y_pred)

        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df is required when intersectional=True")
            labels = self._intersectional_prep(attrs_df, columns)
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                return Result("demographic_parity_difference", np.nan, n_per_group={})
            # Ensure mask is boolean numpy array for proper indexing
            mask = np.asarray(mask, dtype=bool)
            # Use boolean indexing - ensure labels is a proper array
            sens = np.asarray(labels)[mask]
            yp = yp[mask]
        else:
            sens = np.asarray(sensitive)

        # Core metric via adapter (native)
        mr = self._adapter.demographic_parity_difference(
            y_true=None, y_pred=yp, sensitive=sens, min_group_size=self.min_group_size
        )
        res = Result(mr.metric, mr.value, ci=None, effect_size=None, n_per_group=mr.n_per_group)

        # Precompute per-group rates (for CI / effect size)
        groups = [g for g, n in (res.n_per_group or {}).items() if n >= self.min_group_size]
        rates_dict = {}
        for g in groups:
            m = (sens == g) if not isinstance(g, str) else (sens.astype(str) == g)
            # cast sens to str for consistent comparison when labels are categorical-like
            if sens.dtype.kind not in {"U", "S", "O"}:
                m = sens == g
            rates_dict[str(g)] = float(yp[m].mean())

        # CI via bootstrap: resample within each group
        if with_ci and len(groups) >= 2 and np.isfinite(res.value):
            if ci_samples <= 0:
                raise ValueError(
                    "ci_samples must be positive when requesting confidence intervals."
                )
            idx_by_group = {
                g: np.where(
                    (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g
                )[0]
                for g in groups
            }

            def stat_fn(_):
                rates = []
                for g in groups:
                    idx = idx_by_group[g]
                    if idx.size == 0:
                        continue
                    draw = idx[np.random.randint(0, idx.size, size=idx.size)]
                    rates.append(float(yp[draw].mean()))
                return np.nan if len(rates) < 2 else (max(rates) - min(rates))

            dummy = np.arange(sum(len(v) for v in idx_by_group.values()))
            res.ci = bootstrap_ci(dummy, stat_fn, B=ci_samples, level=ci_level, method=ci_method)

        # Effect size: risk ratio of max-rate/min-rate
        if with_effect_size and len(rates_dict) >= 2:
            rmax = max(rates_dict.values())
            rmin = min(rates_dict.values())
            res.effect_size = risk_ratio(rmax, rmin)

        return res

    # ---------- EODD ----------

    def equalized_odds_difference(
        self,
        y_true,
        y_pred,
        sensitive,
        *,
        intersectional: bool = False,
        attrs_df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        with_ci: bool = True,
        ci_level: float = 0.95,
        ci_method: str = "percentile",
        ci_samples: int = 1000,
        with_effect_size: bool = True,  # note: effect size less canonical here; we omit or set None
    ):
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df is required when intersectional=True")
            labels = self._intersectional_prep(attrs_df, columns)
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                return Result("equalized_odds_difference", np.nan, n_per_group={})
            # Ensure mask is boolean numpy array for proper indexing
            mask = np.asarray(mask, dtype=bool)
            # Use boolean indexing - ensure labels is a proper array
            sens = np.asarray(labels)[mask]
            yt = yt[mask]
            yp = yp[mask]
        else:
            sens = np.asarray(sensitive)

        mr = self._adapter.equalized_odds_difference(
            y_true=yt, y_pred=yp, sensitive=sens, min_group_size=self.min_group_size
        )
        res = Result(mr.metric, mr.value, ci=None, effect_size=None, n_per_group=mr.n_per_group)

        # For CI, we need to recompute TPR/FPR per resample
        groups = [g for g, n in (res.n_per_group or {}).items() if n >= self.min_group_size]
        tprs: List[float] = []
        fprs: List[float] = []
        for g in groups:
            idx = np.where(
                (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g
            )[0]
            if idx.size == 0:
                continue
            yt_g = yt[idx]
            yp_g = yp[idx]
            pos = yt_g == 1
            neg = yt_g == 0
            tpr = float((yp_g[pos] == 1).mean()) if pos.any() else np.nan
            fpr = float((yp_g[neg] == 1).mean()) if neg.any() else np.nan
            tprs.append(tpr)
            fprs.append(fpr)

        if with_ci and len(groups) >= 2 and np.isfinite(res.value):
            if ci_samples <= 0:
                raise ValueError(
                    "ci_samples must be positive when requesting confidence intervals."
                )
            idx_by_group = {
                g: np.where(
                    (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g
                )[0]
                for g in groups
            }

            def stat_fn(_):
                tprs, fprs = [], []
                for g in groups:
                    idx = idx_by_group[g]
                    if idx.size == 0:
                        continue
                    draw = idx[np.random.randint(0, idx.size, size=idx.size)]
                    yt_g = yt[draw]
                    yp_g = yp[draw]
                    pos = yt_g == 1
                    neg = yt_g == 0
                    tpr_g = np.nan if pos.sum() == 0 else float((yp_g[pos] == 1).mean())
                    fpr_g = np.nan if neg.sum() == 0 else float((yp_g[neg] == 1).mean())
                    if np.isfinite(tpr_g):
                        tprs.append(tpr_g)
                    if np.isfinite(fpr_g):
                        fprs.append(fpr_g)
                tpr_gap = np.nan if len(tprs) < 2 else (max(tprs) - min(tprs))
                fpr_gap = np.nan if len(fprs) < 2 else (max(fprs) - min(fprs))
                if not np.isfinite(tpr_gap) and not np.isfinite(fpr_gap):
                    return np.nan
                return np.nanmax([tpr_gap, fpr_gap])

            dummy = np.arange(sum(len(v) for v in idx_by_group.values()))
            res.ci = bootstrap_ci(dummy, stat_fn, B=ci_samples, level=ci_level, method=ci_method)

        if with_effect_size:
            ratios: List[float] = []

            def _max_ratio(values: List[float]) -> Optional[float]:
                vals = [v for v in values if np.isfinite(v) and v > 0]
                if len(vals) < 2:
                    return None
                hi, lo = max(vals), min(vals)
                if lo == 0.0:
                    return None
                return hi / lo

            for candidate in (_max_ratio(tprs), _max_ratio(fprs)):
                if candidate is not None:
                    ratios.append(candidate)
            res.effect_size = max(ratios) if ratios else None

        return res

    # ---------- MAE parity (regression) ----------

    def mae_parity_difference(
        self,
        y_true,
        y_pred,
        sensitive,
        *,
        intersectional: bool = False,
        attrs_df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        with_ci: bool = True,
        ci_level: float = 0.95,
        ci_method: str = "percentile",
        ci_samples: int = 1000,
        with_effect_size: bool = True,  # If desired, Cohen's d on absolute errors pairwise is possible
    ):
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df is required when intersectional=True")
            labels = self._intersectional_prep(attrs_df, columns)
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                return Result("mae_parity_difference", np.nan, n_per_group={})
            # Ensure mask is boolean numpy array for proper indexing
            mask = np.asarray(mask, dtype=bool)
            # Use boolean indexing - ensure labels is a proper array
            sens = np.asarray(labels)[mask]
            yt = yt[mask]
            yp = yp[mask]
        else:
            sens = np.asarray(sensitive)

        mr = self._adapter.mae_parity_difference(
            y_true=yt, y_pred=yp, sensitive=sens, min_group_size=self.min_group_size
        )
        res = Result(mr.metric, mr.value, ci=None, effect_size=None, n_per_group=mr.n_per_group)

        groups = [g for g, n in (res.n_per_group or {}).items() if n >= self.min_group_size]
        abs_err = np.abs(yt - yp)

        if with_ci and len(groups) >= 2 and np.isfinite(res.value):
            if ci_samples <= 0:
                raise ValueError(
                    "ci_samples must be positive when requesting confidence intervals."
                )
            idx_by_group = {
                g: np.where(
                    (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g
                )[0]
                for g in groups
            }

            def stat_fn(_):
                maes = []
                for g in groups:
                    idx = idx_by_group[g]
                    if idx.size == 0:
                        continue
                    draw = idx[np.random.randint(0, idx.size, size=idx.size)]
                    maes.append(float(abs_err[draw].mean()))
                return np.nan if len(maes) < 2 else (max(maes) - min(maes))

            dummy = np.arange(sum(len(v) for v in idx_by_group.values()))
            res.ci = bootstrap_ci(dummy, stat_fn, B=ci_samples, level=ci_level, method=ci_method)

        # (Optional) A continuous effect size could be Cohen's d between extreme groups' absolute errors.
        # We omit by default to avoid arbitrary group pair choices; set with_effect_size=True to compute:
        if with_effect_size and len(groups) >= 2:
            # choose extreme groups by MAE
            maes_by_group = {}
            for g in groups:
                idx = np.where(
                    (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g
                )[0]
                maes_by_group[g] = float(abs_err[idx].mean())
            g_max = max(maes_by_group, key=maes_by_group.get)
            g_min = min(maes_by_group, key=maes_by_group.get)
            x = abs_err[
                np.where(
                    (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g_max
                )[0]
            ]
            y = abs_err[
                np.where(
                    (sens.astype(str) if sens.dtype.kind not in {"U", "S", "O"} else sens) == g_min
                )[0]
            ]
            res.effect_size = cohens_d(x, y)

        return res
