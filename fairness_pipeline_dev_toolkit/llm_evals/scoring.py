"""Shared group-rate disparity helpers for Phase 2 LLM evaluators."""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.stats.gap_intervals import binary_gap_interval

from .probes.counterfactual import NEGATIVE_WORDS, REFUSAL_PHRASES

TOXICITY_CI_NOTE = (
    "undefined:no_calibrated_interval (no toxicity-gap interval passed Wave 3a "
    "calibration; see https://github.com/JobCollins/fairness_pipeline_dev_toolkit/issues/63)"
)


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
    """Simultaneous M2a interval for a max−min gap of binary group rates.

    Ignores template pairing (Wave 3a: M2a passed decision 8 on the
    rate-disparity grid; the paired-t candidate did not). Analytic: ``bootstrap_B``
    is irrelevant.
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


def bootstrap_rate_disparity(
    scores: Dict[str, List[float]],
    *,
    B: int = 200,
    level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Deprecated. Prefer :func:`binary_rate_gap_interval` for 0/1 rates.

    Within-group percentile bootstrap of the max−min gap. Under-covers at
    equality (Wave 3a / BL-014); kept for one release as a compatibility path.
    """
    warnings.warn(
        "bootstrap_rate_disparity is deprecated and will be removed in a future "
        "release; for binary rates use binary_rate_gap_interval (simultaneous "
        "Agresti–Caffo). Toxicity has no calibrated default CI "
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
