from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import BootstrapUndefinedError, IntervalUndefinedError
from ..stats.bootstrap import _percentile_ci, bca_ci, stratified_bootstrap_replicates
from ..stats.effect_size import cohens_d, risk_ratio
from ..stats.gap_intervals import (
    binary_gap_interval,
    equalized_odds_gap_interval,
    permutation_gap_pvalue,
)
from ..utils.array_utils import to_numpy_1d
from ..utils.intersectional import build_intersectional_labels, min_group_mask
from .aequitas_adapter import AequitasAdapter
from .base import MetricAdapter, MetricResult
from .eod_undefined import (
    analysed_groups_have_empty_label_stratum,
    empty_label_stratum_ci_note,
)
from .fairlearn_adapter import FairlearnAdapter
from .input_validation import (
    LengthMismatchError,
    check_pandas_indices_aligned,
    nonfinite_drop_caveat,
    prepare_binary_classifier_inputs,
    prepare_regression_metric_inputs,
)
from .native_adapter import NativeAdapter


def _sens_keys(sens: np.ndarray) -> np.ndarray:
    """Stable string group keys aligned with ``sens`` for bootstrap membership tests."""
    return np.asarray(sens, dtype=str)


CI_METHODS = ("simultaneous", "percentile", "bca")

#: BCa refuses below this many rows in any resampling stratum. With fewer rows a
#: group's resampled rate takes only a handful of lattice values, so z0 and the
#: jackknife acceleration are not meaningful; in the BL-031 simulation under
#: ``investigations/wave3a/`` stratified BCa covered 0.00–0.92 (and was undefined
#: at 1 row) for minority groups of 1–5 rows.
BCA_MIN_STRATUM_SIZE = 10
CI_KIND_BY_METHOD = {
    "simultaneous": "simultaneous_pairwise",
    "percentile": "percentile",
    "bca": "bca",
}


def _mae_no_calibrated_interval() -> Tuple[float, float]:
    # Bonferroni Welch-t, a studentized bootstrap and an Edgeworth-corrected Welch all
    # covered a zero MAE gap well below 0.93 for skewed errors in small groups.
    raise IntervalUndefinedError(
        "no_calibrated_interval",
        "no simultaneous MAE-gap interval passed calibration for skewed errors in small "
        "groups; see https://github.com/JobCollins/fairness_pipeline_dev_toolkit/issues/61",
    )


def _validate_ci_method(ci_method: str, ci_samples: int, with_ci: bool) -> None:
    if ci_method not in CI_METHODS:
        raise ValueError(f"Unknown ci_method {ci_method!r}; expected one of {CI_METHODS}")
    if not with_ci:
        return
    if ci_method == "bca":
        warnings.warn(
            "ci_method='bca' is deprecated for gap metrics and will be removed in a "
            "future release: BCa is not calibrated at equality for max−min gaps. "
            "Use the default ci_method='simultaneous'.",
            FutureWarning,
            stacklevel=3,
        )
    if ci_method != "simultaneous" and ci_samples <= 0:
        raise ValueError("ci_samples must be positive when requesting confidence intervals.")


def _set_ci(
    res: "Result",
    *,
    ci_method: str,
    analytic: Callable[[], Tuple[float, float]],
    strata: Callable[[], List[np.ndarray]],
    stat_fn: Callable[[np.ndarray], float],
    ci_samples: int,
    ci_level: float,
    random_state: Optional[int],
) -> None:
    """Fill ``ci`` / ``ci_kind`` / ``ci_note`` on ``res`` under the refuse policy."""
    try:
        if ci_method == "simultaneous":
            ci = analytic()
        else:
            parts = strata()
            if any(p.size == 0 for p in parts):
                raise IntervalUndefinedError(
                    "empty_label_stratum", "a group has no rows in a resampling stratum"
                )
            smallest = min(p.size for p in parts)
            if ci_method == "bca" and smallest < BCA_MIN_STRATUM_SIZE:
                raise BootstrapUndefinedError(
                    "stratum_too_small_for_bca",
                    f"smallest resampling stratum has {smallest} rows < {BCA_MIN_STRATUM_SIZE}",
                )
            boot = stratified_bootstrap_replicates(
                parts, stat_fn, B=ci_samples, random_state=random_state
            )
            if ci_method == "percentile":
                ci = _percentile_ci(boot, ci_level)
            else:
                ci = bca_ci(np.concatenate(parts), stat_fn, boot, level=ci_level)
    except IntervalUndefinedError as err:
        res.ci, res.ci_kind, res.ci_note = None, None, err.ci_note
        return
    res.ci = (float(ci[0]), float(ci[1]))
    res.ci_kind = CI_KIND_BY_METHOD[ci_method]


def _set_pvalue(
    res: "Result",
    values: np.ndarray,
    idx_by_group: List[np.ndarray],
    *,
    strata_values: Optional[np.ndarray] = None,
    n_permutations: int,
    random_state: Optional[int],
) -> None:
    rows = np.concatenate(idx_by_group)
    codes = np.concatenate([np.full(ix.size, k) for k, ix in enumerate(idx_by_group)])
    strata = None if strata_values is None else strata_values[rows]
    try:
        res.p_value = permutation_gap_pvalue(
            values[rows],
            codes,
            strata=strata,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    except IntervalUndefinedError:
        res.p_value = None


def _undefined_ci_note(groups: Sequence[Any], value: float) -> str:
    if len(groups) < 2:
        return f"undefined:too_few_groups ({len(groups)} group(s) meet min_group_size)"
    return "undefined:undefined_value (the point estimate is not finite)"


def dpd_stat_from_indices(
    sample_idx: np.ndarray,
    y_pred: np.ndarray,
    group_of: np.ndarray,
    groups: Sequence[str],
) -> float:
    """Max−min of group means over a bootstrap sample of observation indices.

    Deterministic in ``sample_idx``. Returns ``nan`` if any group in ``groups``
    has zero members in the resample — the estimand is defined on that fixed
    group set, so an incomplete resample does not estimate it.
    """
    idxs = np.asarray(sample_idx, dtype=int)
    rates: List[float] = []
    for g in groups:
        sel = idxs[group_of[idxs] == g]
        if sel.size == 0:
            return float("nan")
        rates.append(float(y_pred[sel].mean()))
    return float(max(rates) - min(rates))


def eod_stat_from_indices(
    sample_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_of: np.ndarray,
    groups: Sequence[str],
) -> float:
    """Equalized-odds gap (max of TPR/FPR gaps) over a bootstrap index sample.

    Deterministic in ``sample_idx``. Returns ``nan`` if any analysis group is
    absent from the resample.
    """
    idxs = np.asarray(sample_idx, dtype=int)
    tprs: List[float] = []
    fprs: List[float] = []
    for g in groups:
        sel = idxs[group_of[idxs] == g]
        if sel.size == 0:
            return float("nan")
        yt_g = y_true[sel]
        yp_g = y_pred[sel]
        pos = yt_g == 1
        neg = yt_g == 0
        tpr_g = np.nan if not np.any(pos) else float((yp_g[pos] == 1).mean())
        fpr_g = np.nan if not np.any(neg) else float((yp_g[neg] == 1).mean())
        if np.isfinite(tpr_g):
            tprs.append(tpr_g)
        if np.isfinite(fpr_g):
            fprs.append(fpr_g)
    tpr_gap = np.nan if len(tprs) < 2 else (max(tprs) - min(tprs))
    fpr_gap = np.nan if len(fprs) < 2 else (max(fprs) - min(fprs))
    if not np.isfinite(tpr_gap) and not np.isfinite(fpr_gap):
        return float("nan")
    return float(np.nanmax([tpr_gap, fpr_gap]))


def mae_gap_stat_from_indices(
    sample_idx: np.ndarray,
    abs_err: np.ndarray,
    group_of: np.ndarray,
    groups: Sequence[str],
) -> float:
    """Max−min of per-group mean absolute error over a bootstrap index sample.

    Deterministic in ``sample_idx``. Returns ``nan`` if any analysis group is
    absent from the resample.
    """
    idxs = np.asarray(sample_idx, dtype=int)
    maes: List[float] = []
    for g in groups:
        sel = idxs[group_of[idxs] == g]
        if sel.size == 0:
            return float("nan")
        maes.append(float(abs_err[sel].mean()))
    return float(max(maes) - min(maes))


@dataclass
class Result(MetricResult):
    """Analyzer result. Field-identical to :class:`~.base.MetricResult` (incl. ``caveat``)."""


class FairnessAnalyzer:
    """
    User-facing orchestrator. Adds:
    - Intersectional grouping
    - min_group_size filtering
    - Optional bootstrap CIs (percentile/BCa)
    - Optional effect sizes (risk ratio for rates, Cohen's d for errors)

    The default ``backend`` is always ``"native"`` (``backend=None`` means native).
    Fairlearn and Aequitas are explicit opt-ins via ``backend="fairlearn"`` /
    ``backend="aequitas"`` (requires ``pip install fairpipe[adapters]``).
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

        self._adapters: Dict[str, MetricAdapter] = {
            "fairlearn": FairlearnAdapter(),
            "aequitas": AequitasAdapter(),
            "native": NativeAdapter(),
        }
        # Default backend is always native (BL-027). Fairlearn / Aequitas are
        # explicit opt-ins only — installing an optional extra must not change results.
        if backend is None:
            backend = "native"
        if backend not in self._adapters:
            raise ValueError(f"Unknown backend: {backend}")
        a = self._adapters[backend]
        if hasattr(a, "available") and not a.available():
            raise RuntimeError(f"Requested backend '{backend}' is not available")
        self._backend = backend
        self._adapter: MetricAdapter = a

        self._cache: Dict[str, Any] = {}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        y_pred_col: str,
        sensitive_col: str,
        y_true_col: Optional[str] = None,
        y_score_col: Optional[str] = None,
        min_group_size: int = 30,
        backend: str = "native",
    ) -> "FairnessAnalyzerDataFrameProxy":
        """Create a proxy bound to a DataFrame, so metric calls need no column args.

        Raises KeyError if any specified column is not present in *df*.
        """
        for col in filter(None, [y_pred_col, sensitive_col, y_true_col, y_score_col]):
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
        analyzer = cls(min_group_size=min_group_size, backend=backend)
        return FairnessAnalyzerDataFrameProxy(
            analyzer, df, y_pred_col, sensitive_col, y_true_col, y_score_col
        )

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
        ci_method: str = "simultaneous",
        ci_samples: int = 2000,
        with_effect_size: bool = True,
        with_pvalue: Optional[bool] = None,
        n_permutations: int = 2000,
        random_state: Optional[int] = 42,
    ):
        """Demographic parity difference: max − min positive-prediction rate over groups.

        Parameters
        ----------
        y_pred, sensitive
            Binary predictions and group labels. Groups smaller than
            ``min_group_size`` are excluded from the value, CI and p-value.
        with_ci
            Compute a confidence interval for the gap (default ``True``).
        ci_level
            Confidence level (default 0.95).
        ci_method
            ``"simultaneous"`` (default): Bonferroni Agresti–Caffo intervals for every
            pairwise rate difference, inverted into an interval for the gap
            (``ci_kind="simultaneous_pairwise"``). Analytic; covers a true gap of 0.
            ``"percentile"``: within-group stratified bootstrap percentile interval.
            **Not calibrated at equality** (BL-014: coverage of a true gap of 0 is
            ≈0 with three or more groups); explicit opt-in only.
            ``"bca"``: deprecated for gap metrics (``FutureWarning``), same
            calibration problem; refuses (``ci=None`` + ``ci_note``) rather than
            returning NaN.
        ci_samples
            Bootstrap replicates for ``"percentile"`` / ``"bca"`` (default 2000).
            Ignored by ``"simultaneous"``.
        with_effect_size
            Risk ratio of the highest to the lowest group rate.
        with_pvalue
            Permutation-test ``p_value`` for "all group rates equal" (gap statistic,
            group labels shuffled). ``None`` (default) follows ``with_ci``.
        n_permutations
            Label shuffles for the p-value (default 2000).
        random_state
            Seed for bootstrap and permutation draws (default 42).

        Returns
        -------
        Result
            ``ci`` is a finite ``(lower, upper)`` or ``None``; when a requested CI is
            undefined, ``ci_note`` starts with ``"undefined:<reason>"``. Only
            ``p_value`` supports a "significant" claim; ``ci[1] < δ`` reads as "below
            δ with ``ci_level`` confidence".

        Examples
        --------
        >>> fa = FairnessAnalyzer(min_group_size=5, backend="native")
        >>> r = fa.demographic_parity_difference([1, 0] * 10, ["a"] * 10 + ["b"] * 10)
        >>> r.ci_kind
        'simultaneous_pairwise'
        """
        _validate_ci_method(ci_method, ci_samples, with_ci)
        want_p = with_ci if with_pvalue is None else with_pvalue
        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df is required when intersectional=True")
            check_pandas_indices_aligned(y_pred=y_pred, attrs_df=attrs_df)
        else:
            check_pandas_indices_aligned(y_pred=y_pred, sensitive=sensitive)

        yp = to_numpy_1d(y_pred, "y_pred")

        if intersectional:
            assert attrs_df is not None  # checked above
            if len(yp) != len(attrs_df):
                raise LengthMismatchError(
                    f"y_pred and attrs_df must have the same length; "
                    f"got y_pred={len(yp)}, attrs_df={len(attrs_df)}. "
                    "Align or truncate before calling the metric."
                )
            labels = self._intersectional_prep(attrs_df, columns)
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                return Result(
                    "demographic_parity_difference",
                    np.nan,
                    n_per_group={},
                    ci_note=_undefined_ci_note([], np.nan) if with_ci else None,
                )
            # Ensure mask is boolean numpy array for proper indexing
            mask = np.asarray(mask, dtype=bool)
            # Use boolean indexing - ensure labels is a proper array
            sens = np.asarray(labels)[mask]
            yp = yp[mask]
        else:
            sens = to_numpy_1d(sensitive, "sensitive")

        prepared = prepare_binary_classifier_inputs(
            y_pred=yp, sensitive=sens, y_true=None, require_y_true=False
        )
        yp, sens = prepared.y_pred, prepared.sensitive
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)

        # Core metric via adapter (native)
        mr = self._adapter.demographic_parity_difference(
            y_true=None, y_pred=yp, sensitive=sens, min_group_size=self.min_group_size
        )
        res = Result(
            mr.metric,
            mr.value,
            ci=None,
            effect_size=None,
            n_per_group=mr.n_per_group,
            caveat=drop_caveat or mr.caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )

        # Precompute per-group rates (for CI / effect size)
        groups = [g for g, n in (res.n_per_group or {}).items() if n >= self.min_group_size]
        rates_dict = {}
        for g in groups:
            m = (sens == g) if not isinstance(g, str) else (sens.astype(str) == g)
            # cast sens to str for consistent comparison when labels are categorical-like
            if sens.dtype.kind not in {"U", "S", "O"}:
                m = sens == g
            rates_dict[str(g)] = float(yp[m].mean())

        estimable = len(groups) >= 2 and np.isfinite(res.value)
        if with_ci and not estimable:
            res.ci_note = _undefined_ci_note(groups, res.value)
        if estimable and (with_ci or want_p):
            group_keys = [str(g) for g in groups]
            group_of = _sens_keys(sens)
            idx_by_group = [np.flatnonzero(group_of == k) for k in group_keys]
            yp_f = np.asarray(yp, dtype=float)

            def stat_fn(sample_idx):
                return dpd_stat_from_indices(sample_idx, yp_f, group_of, group_keys)

            if with_ci:
                _set_ci(
                    res,
                    ci_method=ci_method,
                    analytic=lambda: binary_gap_interval(
                        [yp_f[ix].sum() for ix in idx_by_group],
                        [ix.size for ix in idx_by_group],
                        ci_level,
                    ),
                    strata=lambda: idx_by_group,
                    stat_fn=stat_fn,
                    ci_samples=ci_samples,
                    ci_level=ci_level,
                    random_state=random_state,
                )
            if want_p:
                _set_pvalue(
                    res,
                    yp_f,
                    idx_by_group,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )

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
        ci_method: str = "simultaneous",
        ci_samples: int = 2000,
        with_effect_size: bool = True,  # note: effect size less canonical here; we omit or set None
        with_pvalue: Optional[bool] = None,
        n_permutations: int = 2000,
        random_state: Optional[int] = 42,
    ):
        """Equalized odds difference: ``max(TPR gap, FPR gap)`` over groups.

        CI arguments, ``with_pvalue``, ``n_permutations`` and ``random_state`` behave
        as in :meth:`demographic_parity_difference`, with these differences:

        - ``"simultaneous"`` uses Agresti–Caffo intervals for every pairwise TPR
          *and* FPR difference under one Bonferroni correction; the gap interval is
          the max over both families.
        - Bootstrap methods resample within group × ``y_true`` strata, so every
          replicate keeps each group's positives and negatives.
        - The permutation test shuffles group labels within ``y_true`` strata.
        - If an analysed group has no positives or no negatives, equalized odds is
          **undefined** on every backend: ``value`` is NaN, ``ci`` is ``None`` with
          ``ci_note="undefined:empty_label_stratum (...)"``, and ``p_value`` is ``None``.
          Gating reports this as undefined (not pass or fail).

        Examples
        --------
        >>> fa = FairnessAnalyzer(min_group_size=5, backend="native")
        >>> r = fa.equalized_odds_difference(
        ...     [1, 0] * 10, [1, 0, 0, 1] * 5, ["a"] * 10 + ["b"] * 10
        ... )
        >>> r.ci_kind
        'simultaneous_pairwise'
        """
        _validate_ci_method(ci_method, ci_samples, with_ci)
        want_p = with_ci if with_pvalue is None else with_pvalue
        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df is required when intersectional=True")
            check_pandas_indices_aligned(y_true=y_true, y_pred=y_pred, attrs_df=attrs_df)
        else:
            check_pandas_indices_aligned(y_true=y_true, y_pred=y_pred, sensitive=sensitive)

        yt = to_numpy_1d(y_true, "y_true")
        yp = to_numpy_1d(y_pred, "y_pred")

        if intersectional:
            assert attrs_df is not None  # checked above
            if len(yp) != len(attrs_df) or len(yt) != len(attrs_df):
                raise LengthMismatchError(
                    f"y_true, y_pred, and attrs_df must have the same length; "
                    f"got y_true={len(yt)}, y_pred={len(yp)}, attrs_df={len(attrs_df)}. "
                    "Align or truncate before calling the metric."
                )
            labels = self._intersectional_prep(attrs_df, columns)
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                return Result(
                    "equalized_odds_difference",
                    np.nan,
                    n_per_group={},
                    ci_note=_undefined_ci_note([], np.nan) if with_ci else None,
                )
            # Ensure mask is boolean numpy array for proper indexing
            mask = np.asarray(mask, dtype=bool)
            # Use boolean indexing - ensure labels is a proper array
            sens = np.asarray(labels)[mask]
            yt = yt[mask]
            yp = yp[mask]
        else:
            sens = to_numpy_1d(sensitive, "sensitive")

        prepared = prepare_binary_classifier_inputs(
            y_pred=yp, sensitive=sens, y_true=yt, require_y_true=True
        )
        yp, sens = prepared.y_pred, prepared.sensitive
        assert prepared.y_true is not None  # require_y_true=True
        yt = prepared.y_true
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)

        mr = self._adapter.equalized_odds_difference(
            y_true=yt, y_pred=yp, sensitive=sens, min_group_size=self.min_group_size
        )
        res = Result(
            mr.metric,
            mr.value,
            ci=None,
            effect_size=None,
            n_per_group=mr.n_per_group,
            caveat=drop_caveat or mr.caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )

        # For CI / undefined detection we use string group keys matching n_per_group.
        groups = [g for g, n in (res.n_per_group or {}).items() if n >= self.min_group_size]
        group_of = _sens_keys(sens)
        empty_stratum = (
            analysed_groups_have_empty_label_stratum(yt, group_of, groups) if groups else False
        )
        if empty_stratum:
            res.value = float("nan")
            res.ci = None
            res.ci_kind = None
            res.ci_note = empty_label_stratum_ci_note()
            res.p_value = None
            if with_effect_size:
                res.effect_size = None
            return res

        tprs: List[float] = []
        fprs: List[float] = []
        for g in groups:
            idx = np.where(group_of == g)[0]
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

        estimable = len(groups) >= 2 and np.isfinite(res.value)
        if with_ci and not estimable:
            res.ci_note = _undefined_ci_note(groups, res.value)
        if estimable and (with_ci or want_p):
            group_keys = [str(g) for g in groups]
            idx_by_group = [np.flatnonzero(group_of == k) for k in group_keys]
            yt_f = np.asarray(yt, dtype=float)
            yp_f = np.asarray(yp, dtype=float)
            pos_idx = [ix[yt_f[ix] == 1] for ix in idx_by_group]
            neg_idx = [ix[yt_f[ix] == 0] for ix in idx_by_group]

            def stat_fn(sample_idx):
                return eod_stat_from_indices(sample_idx, yt_f, yp_f, group_of, group_keys)

            if with_ci:
                _set_ci(
                    res,
                    ci_method=ci_method,
                    analytic=lambda: equalized_odds_gap_interval(
                        [yp_f[ix].sum() for ix in pos_idx],
                        [ix.size for ix in pos_idx],
                        [yp_f[ix].sum() for ix in neg_idx],
                        [ix.size for ix in neg_idx],
                        ci_level,
                    ),
                    strata=lambda: pos_idx + neg_idx,
                    stat_fn=stat_fn,
                    ci_samples=ci_samples,
                    ci_level=ci_level,
                    random_state=random_state,
                )
            if want_p:
                _set_pvalue(
                    res,
                    yp_f,
                    idx_by_group,
                    strata_values=yt_f.astype(np.int64),
                    n_permutations=n_permutations,
                    random_state=random_state,
                )

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
        ci_method: str = "simultaneous",
        ci_samples: int = 2000,
        with_effect_size: bool = True,  # If desired, Cohen's d on absolute errors pairwise is possible
        with_pvalue: Optional[bool] = None,
        n_permutations: int = 2000,
        random_state: Optional[int] = 42,
    ):
        """MAE parity difference: max − min mean absolute error over groups.

        CI arguments, ``with_pvalue``, ``n_permutations`` and ``random_state`` behave
        as in :meth:`demographic_parity_difference`, with these differences:

        - ``"simultaneous"`` (default) returns ``ci=None`` with
          ``ci_note="undefined:no_calibrated_interval (...)"``. No candidate
          interval (Bonferroni Welch-t, studentized bootstrap, Edgeworth-corrected
          Welch) reached the required coverage of a zero gap for skewed errors in
          small groups (issue #61). :func:`~fairness_pipeline_dev_toolkit.stats.
          gap_intervals.welch_gap_interval` is available if you have checked it for
          your data.
        - ``"percentile"`` is an uncalibrated opt-in, as for DPD.
        - The permutation ``p_value`` shuffles group labels over absolute errors; its
          type-I error was ≤ 0.06 in the same simulation, so it is reported.

        Examples
        --------
        >>> fa = FairnessAnalyzer(min_group_size=5, backend="native")
        >>> r = fa.mae_parity_difference(
        ...     [0.0] * 20, [0.1, 0.3] * 5 + [0.2, 0.5] * 5, ["a"] * 10 + ["b"] * 10
        ... )
        >>> r.ci is None, r.ci_note.split(" ")[0]
        (True, 'undefined:no_calibrated_interval')
        >>> 0 < r.p_value <= 1
        True
        """
        _validate_ci_method(ci_method, ci_samples, with_ci)
        want_p = with_ci if with_pvalue is None else with_pvalue
        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df is required when intersectional=True")
            check_pandas_indices_aligned(y_true=y_true, y_pred=y_pred, attrs_df=attrs_df)
        else:
            check_pandas_indices_aligned(y_true=y_true, y_pred=y_pred, sensitive=sensitive)

        yt = to_numpy_1d(y_true, "y_true")
        yp = to_numpy_1d(y_pred, "y_pred")

        if intersectional:
            assert attrs_df is not None  # checked above
            if len(yp) != len(attrs_df) or len(yt) != len(attrs_df):
                raise LengthMismatchError(
                    f"y_true, y_pred, and attrs_df must have the same length; "
                    f"got y_true={len(yt)}, y_pred={len(yp)}, attrs_df={len(attrs_df)}. "
                    "Align or truncate before calling the metric."
                )
            labels = self._intersectional_prep(attrs_df, columns)
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                return Result(
                    "mae_parity_difference",
                    np.nan,
                    n_per_group={},
                    ci_note=_undefined_ci_note([], np.nan) if with_ci else None,
                )
            # Ensure mask is boolean numpy array for proper indexing
            mask = np.asarray(mask, dtype=bool)
            # Use boolean indexing - ensure labels is a proper array
            sens = np.asarray(labels)[mask]
            yt = yt[mask]
            yp = yp[mask]
        else:
            sens = to_numpy_1d(sensitive, "sensitive")

        prepared = prepare_regression_metric_inputs(y_true=yt, y_pred=yp, sensitive=sens)
        yp, sens = prepared.y_pred, prepared.sensitive
        assert prepared.y_true is not None
        yt = prepared.y_true
        drop_caveat = nonfinite_drop_caveat(prepared.n_dropped_nonfinite)

        mr = self._adapter.mae_parity_difference(
            y_true=yt, y_pred=yp, sensitive=sens, min_group_size=self.min_group_size
        )
        res = Result(
            mr.metric,
            mr.value,
            ci=None,
            effect_size=None,
            n_per_group=mr.n_per_group,
            caveat=drop_caveat or mr.caveat,
            n_dropped_nonfinite=prepared.n_dropped_nonfinite,
        )

        groups = [g for g, n in (res.n_per_group or {}).items() if n >= self.min_group_size]
        abs_err = np.abs(yt - yp)

        estimable = len(groups) >= 2 and np.isfinite(res.value)
        if with_ci and not estimable:
            res.ci_note = _undefined_ci_note(groups, res.value)
        if estimable and (with_ci or want_p):
            group_keys = [str(g) for g in groups]
            group_of = _sens_keys(sens)
            idx_by_group = [np.flatnonzero(group_of == k) for k in group_keys]
            abs_err_f = np.asarray(abs_err, dtype=float)

            def stat_fn(sample_idx):
                return mae_gap_stat_from_indices(sample_idx, abs_err_f, group_of, group_keys)

            if with_ci:
                _set_ci(
                    res,
                    ci_method=ci_method,
                    analytic=_mae_no_calibrated_interval,
                    strata=lambda: idx_by_group,
                    stat_fn=stat_fn,
                    ci_samples=ci_samples,
                    ci_level=ci_level,
                    random_state=random_state,
                )
            if want_p:
                _set_pvalue(
                    res,
                    abs_err_f,
                    idx_by_group,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )

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
            g_max = max(maes_by_group, key=lambda g: maes_by_group[g])
            g_min = min(maes_by_group, key=lambda g: maes_by_group[g])
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


class FairnessAnalyzerDataFrameProxy:
    """Bound proxy returned by :meth:`FairnessAnalyzer.from_dataframe`.

    Stores a DataFrame and column names so that metric methods can be called
    without repeating column arguments each time.
    """

    def __init__(
        self,
        analyzer: FairnessAnalyzer,
        df: pd.DataFrame,
        y_pred_col: str,
        sensitive_col: str,
        y_true_col: Optional[str] = None,
        y_score_col: Optional[str] = None,
    ) -> None:
        self._analyzer = analyzer
        self._df = df
        self._y_pred_col = y_pred_col
        self._sensitive_col = sensitive_col
        self._y_true_col = y_true_col
        self._y_score_col = y_score_col

    def demographic_parity_difference(self, **kwargs) -> Result:
        return self._analyzer.demographic_parity_difference(
            y_pred=self._df[self._y_pred_col],
            sensitive=self._df[self._sensitive_col],
            **kwargs,
        )

    def equalized_odds_difference(self, **kwargs) -> Result:
        if self._y_true_col is None:
            raise ValueError(
                "y_true_col must be specified in from_dataframe() to call "
                "equalized_odds_difference()."
            )
        return self._analyzer.equalized_odds_difference(
            y_true=self._df[self._y_true_col],
            y_pred=self._df[self._y_pred_col],
            sensitive=self._df[self._sensitive_col],
            **kwargs,
        )

    def mae_parity_difference(self, **kwargs) -> Result:
        if self._y_true_col is None:
            raise ValueError(
                "y_true_col must be specified in from_dataframe() to call "
                "mae_parity_difference()."
            )
        return self._analyzer.mae_parity_difference(
            y_true=self._df[self._y_true_col],
            y_pred=self._df[self._y_pred_col],
            sensitive=self._df[self._sensitive_col],
            **kwargs,
        )
