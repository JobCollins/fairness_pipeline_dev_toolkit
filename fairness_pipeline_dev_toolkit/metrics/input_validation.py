"""Shared input validation for classifier (and regression MAE) metrics.

Called from :class:`~fairness_pipeline_dev_toolkit.metrics.core.FairnessAnalyzer`
and from every metric adapter so direct adapter use cannot bypass the contract.

Classifier metrics assume **binary labels encoded as 0 / 1**, with positive
class **1** (see ``POSITIVE_LABEL``). That matches the historical EOD
implementation (``y_true == 1`` / ``y_true == 0``) and DPD selection-rate
means. Multiclass is rejected in this pass — not redefined.

Non-finite ``y_true`` / ``y_pred`` values are **dropped** (with a reported
count), matching protected-attribute ``nan_policy="exclude"``: incomplete
rows are omitted rather than inventing a rate. The drop count is returned on
the prepared-inputs object and must be surfaced on ``MetricResult``.

When **two or more** inputs are pandas ``Series`` / ``Index`` / ``DataFrame``,
their ``.index`` values must be equal (same labels **and** order). Mismatched
indices raise :class:`IndexMismatchError` — fairpipe does not silently zip by
position (which can invert a disparity) and does not auto-align (a silent
result change). A single pandas object mixed with arrays/lists is allowed and
uses positional semantics, because arrays have no index to honor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.array_utils import to_numpy_1d

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


class MetricInputError(ValueError):
    """Base for invalid fairness-metric inputs."""


class LengthMismatchError(MetricInputError):
    """``y_true`` / ``y_pred`` / ``sensitive`` length mismatch."""


class IndexMismatchError(MetricInputError):
    """Pandas inputs have unequal indices (labels or order)."""


class MulticlassNotSupportedError(MetricInputError):
    """More than two distinct label values — multiclass is not supported."""


class NonBinaryEncodingError(MetricInputError):
    """Binary cardinality but values are not {0, 1}."""


@dataclass(frozen=True)
class PreparedMetricInputs:
    """Aligned, cleaned arrays ready for a metric computation."""

    y_pred: np.ndarray
    sensitive: np.ndarray
    y_true: Optional[np.ndarray]
    n_dropped_nonfinite: int
    positive_label: int = POSITIVE_LABEL


def _pandas_index(obj: Any) -> Optional[pd.Index]:
    if isinstance(obj, pd.Series):
        return obj.index
    if isinstance(obj, pd.DataFrame):
        return obj.index
    if isinstance(obj, pd.Index):
        return obj
    return None


def check_pandas_indices_aligned(**named_inputs: Any) -> None:
    """Reject mismatched pandas indices among metric inputs (Wave 1e / BL-020).

    Only runs when two or more arguments are pandas objects with an index.
    Non-pandas inputs (``list``, ``ndarray``) are ignored for this check so a
    mixed Series+array call stays positional — there is no second index to
    disagree with.
    """
    indexed: List[Tuple[str, pd.Index]] = []
    for name, value in named_inputs.items():
        if value is None:
            continue
        idx = _pandas_index(value)
        if idx is not None:
            indexed.append((name, idx))

    if len(indexed) < 2:
        return

    first_name, first_index = indexed[0]
    for name, idx in indexed[1:]:
        if first_index.equals(idx):
            continue
        raise IndexMismatchError(
            f"Pandas inputs {first_name!r} and {name!r} have unequal indices "
            f"(labels or order differ). fairpipe does not zip Series by "
            f"position — that can invert a fairness conclusion — and does not "
            f"auto-align. Align explicitly with .reindex() / .loc / "
            f".reset_index(drop=True) so every input shares one index, or pass "
            f"plain arrays/lists for positional semantics. "
            f"{first_name} index={list(first_index)!r}; {name} index={list(idx)!r}."
        )


def _length_of(name: str, arr: np.ndarray) -> Tuple[str, int]:
    return name, int(arr.shape[0])


def check_metric_lengths(
    *,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> None:
    """Raise :class:`LengthMismatchError` naming each array's length."""
    parts = [_length_of("y_pred", y_pred), _length_of("sensitive", sensitive)]
    if y_true is not None:
        parts.insert(0, _length_of("y_true", y_true))
    lengths = [n for _, n in parts]
    if len(set(lengths)) != 1:
        detail = ", ".join(f"{name}={n}" for name, n in parts)
        raise LengthMismatchError(
            f"y_true, y_pred, and sensitive must have the same length; got {detail}. "
            "Align or truncate arrays before calling the metric."
        )


def _nonfinite_row_mask(arr: np.ndarray) -> np.ndarray:
    """True where a row is missing / non-finite for label or score purposes."""
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    num = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    return ~np.isfinite(num)


def _coerce_binary_01(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce bool/0/1 labels to int {0, 1}; reject other encodings."""
    if arr.size == 0:
        return np.asarray(arr, dtype=int)

    if arr.dtype == bool or (
        isinstance(arr.dtype, np.dtype) and np.issubdtype(arr.dtype, np.bool_)
    ):
        return arr.astype(int)

    num = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(num)):
        raise MetricInputError(
            f"{name} contains non-finite values after cleaning; remove or impute "
            "them before calling the metric."
        )

    uniq = np.unique(num)
    allowed = {float(NEGATIVE_LABEL), float(POSITIVE_LABEL)}
    if len(uniq) > 2:
        raise MulticlassNotSupportedError(
            f"{name} has {len(uniq)} distinct values {sorted(uniq.tolist())[:8]}"
            f"{'…' if len(uniq) > 8 else ''}; "
            "classifier metrics require hard binary labels in {0, 1} with positive "
            f"class {POSITIVE_LABEL}. If you passed probabilities (e.g. "
            "`predict_proba(X)[:, 1]`), threshold them first: "
            "`(proba >= 0.5).astype(int)`. Multiclass outcomes are not supported."
        )
    if not set(uniq.tolist()) <= allowed:
        raise NonBinaryEncodingError(
            f"{name} has distinct values {sorted(uniq.tolist())}, but classifier "
            f"metrics require labels in {{{NEGATIVE_LABEL}, {POSITIVE_LABEL}}} "
            f"with positive class {POSITIVE_LABEL}. Remap encodings "
            f"(e.g. `(y == positive_class).astype(int)`), or if these are "
            "probabilities, threshold first: `(proba >= 0.5).astype(int)`."
        )
    return num.astype(int)


def _drop_nonfinite_rows(
    *,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    y_true: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Drop rows with non-finite y_pred (and y_true if present); return count."""
    drop = _nonfinite_row_mask(y_pred)
    if y_true is not None:
        drop = drop | _nonfinite_row_mask(y_true)
    n_dropped = int(np.count_nonzero(drop))
    keep = ~drop
    yp = y_pred[keep]
    sens = sensitive[keep]
    yt = None if y_true is None else y_true[keep]
    return yp, sens, yt, n_dropped


def prepare_binary_classifier_inputs(
    *,
    y_pred: Any,
    sensitive: Any,
    y_true: Any = None,
    require_y_true: bool = False,
) -> PreparedMetricInputs:
    """Validate and clean inputs for DPD / EOD (binary 0/1, positive=1)."""
    check_pandas_indices_aligned(y_true=y_true, y_pred=y_pred, sensitive=sensitive)

    yp = to_numpy_1d(y_pred, "y_pred")
    sens = to_numpy_1d(sensitive, "sensitive")
    yt: Optional[np.ndarray]
    if y_true is None:
        if require_y_true:
            raise MetricInputError("y_true is required for this metric.")
        yt = None
    else:
        yt = to_numpy_1d(y_true, "y_true")

    check_metric_lengths(y_pred=yp, sensitive=sens, y_true=yt)
    yp, sens, yt, n_dropped = _drop_nonfinite_rows(y_pred=yp, sensitive=sens, y_true=yt)

    yp = _coerce_binary_01(yp, "y_pred")
    if yt is not None:
        yt = _coerce_binary_01(yt, "y_true")

    return PreparedMetricInputs(
        y_pred=yp,
        sensitive=sens,
        y_true=yt,
        n_dropped_nonfinite=n_dropped,
        positive_label=POSITIVE_LABEL,
    )


def prepare_regression_metric_inputs(
    *,
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
) -> PreparedMetricInputs:
    """Validate and clean inputs for MAE parity (continuous; no binary check)."""
    check_pandas_indices_aligned(y_true=y_true, y_pred=y_pred, sensitive=sensitive)

    yt = to_numpy_1d(y_true, "y_true")
    yp = to_numpy_1d(y_pred, "y_pred")
    sens = to_numpy_1d(sensitive, "sensitive")
    check_metric_lengths(y_pred=yp, sensitive=sens, y_true=yt)
    yp, sens, yt, n_dropped = _drop_nonfinite_rows(y_pred=yp, sensitive=sens, y_true=yt)
    yt_f = pd.to_numeric(pd.Series(yt), errors="coerce").to_numpy(dtype=float)
    yp_f = pd.to_numeric(pd.Series(yp), errors="coerce").to_numpy(dtype=float)
    return PreparedMetricInputs(
        y_pred=yp_f,
        sensitive=sens,
        y_true=yt_f,
        n_dropped_nonfinite=n_dropped,
        positive_label=POSITIVE_LABEL,
    )


def nonfinite_drop_caveat(n_dropped: int) -> Optional[str]:
    """Human-readable caveat when rows were dropped for non-finiteness."""
    if n_dropped <= 0:
        return None
    return (
        f"Dropped {n_dropped} row(s) with non-finite y_true/y_pred "
        "(consistent with nan_policy='exclude' on protected attributes). "
        "Impute or filter incomplete labels if those rows should contribute "
        "to the disparity estimate."
    )


def merge_caveats(*parts: Optional[str]) -> Optional[str]:
    texts = [p.strip() for p in parts if p and str(p).strip()]
    if not texts:
        return None
    return " ".join(texts)
