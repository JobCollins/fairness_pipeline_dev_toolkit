"""Shared group-rate disparity helpers for Phase 2 LLM evaluators."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import t as student_t

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.stats.gap_intervals import (
    binary_gap_interval,
    simultaneous_gap_bounds,
)

from .probes.counterfactual import NEGATIVE_WORDS, REFUSAL_PHRASES


def refusal_score(text: str) -> float:
    normalized = (text or "").strip().lower()
    return 1.0 if any(phrase in normalized for phrase in REFUSAL_PHRASES) else 0.0


def toxicity_score(text: str) -> float:
    """Lexical toxicity/negativity proxy — no external moderation API required."""
    words = (text or "").strip().lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in NEGATIVE_WORDS)
    return hits / len(words)


def n_per_group(scores: Dict[str, List[float]]) -> Dict[str, int]:
    return {group: len(values) for group, values in scores.items()}


def rate_disparity(scores: Dict[str, List[float]]) -> float:
    means = [float(np.mean(values)) for values in scores.values() if values]
    if len(means) < 2:
        return float("nan")
    return float(max(means) - min(means))


def binary_rate_gap_interval(
    scores: Dict[str, List[float]],
    *,
    level: float = 0.95,
) -> Tuple[float, float]:
    """Simultaneous Agresti–Caffo interval for a max−min gap of binary group rates.

    Ignores template pairing (calibration evidence in ``investigations/wave3a/``).
    Analytic: ``bootstrap_B`` is irrelevant.
    """
    groups = [g for g, vals in scores.items() if vals]
    if len(groups) < 2:
        raise IntervalUndefinedError("too_few_groups", f"K={len(groups)} < 2")
    successes = []
    ns = []
    for g in groups:
        arr = np.asarray(scores[g], dtype=float)
        if not np.all((arr == 0.0) | (arr == 1.0)):
            raise ValueError("binary_rate_gap_interval requires 0/1 scores")
        successes.append(float(arr.sum()))
        ns.append(float(arr.size))
    lo, hi = binary_gap_interval(successes, ns, level=level)
    point = rate_disparity(scores)
    if np.isfinite(point):
        lo, hi = min(lo, point), max(hi, point)
    return float(lo), float(hi)


def paired_template_gap_interval(
    scores_by_template: Dict[str, Dict[Any, float]],
    *,
    level: float = 0.95,
) -> Tuple[float, float]:
    """Bonferroni-t on per-template group differences, inverted to a gap interval.

    Opt-in helper for ``toxicity_sentiment_disparity``
    (``ci_method="template_bonferroni_t"``). The metric's default CI is undefined:
    the interval could not be validated on real data because the recorded
    fixture scores are all zero (issue #63; see ``investigations/wave3a/``).
    ``scores_by_template`` maps group → ``{replicate_id: score}``. Only templates
    complete for every group are kept.
    """
    groups = [g for g, m in scores_by_template.items() if m]
    if len(groups) < 2:
        raise IntervalUndefinedError("too_few_groups", f"K={len(groups)} < 2")
    complete = set.intersection(*(set(scores_by_template[g]) for g in groups))
    if len(complete) < 2:
        raise IntervalUndefinedError(
            "too_few_templates", f"T={len(complete)} < 2 complete templates"
        )
    reps = sorted(complete)
    mat = np.column_stack([[scores_by_template[g][r] for r in reps] for g in groups])  # (T, K)
    T, k = mat.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    a_tail = (1.0 - level) / (2.0 * len(pairs))
    crit = float(student_t.ppf(1.0 - a_tail, T - 1))
    lo, hi = [], []
    for i, j in pairs:
        d = mat[:, i] - mat[:, j]
        sd = float(d.std(ddof=1))
        if sd <= 0.0:
            raise IntervalUndefinedError("zero_variance", "all template diffs equal")
        se = sd / np.sqrt(T)
        m = float(d.mean())
        lo.append(m - crit * se)
        hi.append(m + crit * se)
    bounds = simultaneous_gap_bounds(lo, hi, max_gap=1.0)
    point = float(mat.mean(axis=0).max() - mat.mean(axis=0).min())
    return min(bounds[0], point), max(bounds[1], point)


def bootstrap_rate_disparity(
    scores: Dict[str, List[float]],
    *,
    B: int = 200,
    level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Deprecated. Prefer :func:`binary_rate_gap_interval` for 0/1 rates.

    Within-group percentile bootstrap of the max−min gap. Under-covers at
    equality (see ``investigations/wave3a/``); kept for one release as a
    compatibility path.
    """
    warnings.warn(
        "bootstrap_rate_disparity is deprecated and will be removed in a future "
        "release; for binary rates use binary_rate_gap_interval (simultaneous "
        "Agresti–Caffo). For toxicity use paired_template_gap_interval "
        "(https://github.com/JobCollins/fairness_pipeline_dev_toolkit/issues/63).",
        FutureWarning,
        stacklevel=2,
    )
    groups = [g for g, vals in scores.items() if vals]
    if len(groups) < 2:
        raise IntervalUndefinedError("too_few_groups", f"K={len(groups)} < 2")
    arrays = {g: np.asarray(scores[g], dtype=float) for g in groups}
    rng = np.random.default_rng(random_state)
    stats = np.empty(B, dtype=float)
    for b in range(B):
        means = []
        for g in groups:
            x = arrays[g]
            sample = x[rng.integers(0, len(x), len(x))]
            means.append(float(np.mean(sample)))
        stats[b] = max(means) - min(means)
    alpha = (1 - level) / 2
    return float(np.percentile(stats, alpha * 100)), float(np.percentile(stats, (1 - alpha) * 100))


def apply_scorer(
    rows: List[Dict[str, str]],
    scorer: Callable[[str], float],
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for row in rows:
        out.setdefault(row["group"], []).append(scorer(row.get("response") or ""))
    return out
