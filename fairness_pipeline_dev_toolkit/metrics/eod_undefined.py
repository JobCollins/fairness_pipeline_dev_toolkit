"""Shared equalized-odds undefined rules (empty label strata)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError

# Machine-readable reason shared by adapters, CIs, and gating.
EMPTY_LABEL_STRATUM = "empty_label_stratum"
EMPTY_LABEL_STRATUM_DETAIL = (
    "every analysed group needs at least one positive and one negative label"
)


def empty_label_stratum_ci_note() -> str:
    """``ci_note`` for EOD when a group lacks positives or negatives."""
    return IntervalUndefinedError(EMPTY_LABEL_STRATUM, EMPTY_LABEL_STRATUM_DETAIL).ci_note


def analysed_groups_have_empty_label_stratum(
    y_true: np.ndarray,
    sensitive: np.ndarray,
    groups: Sequence[object] | None = None,
) -> bool:
    """Return True if any analysed group has no positives or no negatives.

    Parameters
    ----------
    y_true, sensitive :
        Aligned 1-d arrays (already filtered to groups meeting ``min_group_size``).
    groups :
        Optional explicit group labels. Defaults to ``np.unique(sensitive)``.
    """
    yt = np.asarray(y_true)
    s = np.asarray(sensitive)
    if groups is None:
        groups = list(np.unique(s))
    for g in groups:
        mask = s == g
        if not np.any(mask):
            continue
        yt_g = yt[mask]
        if not np.any(yt_g == 1) or not np.any(yt_g == 0):
            return True
    return False


def equalized_odds_point_estimate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    *,
    groups: Iterable[object] | None = None,
) -> float:
    """Max of TPR/FPR gaps, or NaN when any analysed group lacks a label stratum.

    This is the Wave 4 / BL-019 equalized-odds contract: incomplete EO is
    undefined (NaN), never 0.0.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    s = np.asarray(sensitive)
    group_list = list(groups) if groups is not None else list(np.unique(s))
    if len(group_list) < 2:
        return float("nan")
    if analysed_groups_have_empty_label_stratum(yt, s, group_list):
        return float("nan")

    tpr: dict[str, float] = {}
    fpr: dict[str, float] = {}
    for g in group_list:
        m = s == g
        yt_g, yp_g = yt[m], yp[m]
        pos = yt_g == 1
        neg = yt_g == 0
        tpr[str(g)] = float(np.mean(yp_g[pos]))
        fpr[str(g)] = float(np.mean(yp_g[neg]))

    tpr_gap = max(tpr.values()) - min(tpr.values())
    fpr_gap = max(fpr.values()) - min(fpr.values())
    return float(max(tpr_gap, fpr_gap))
