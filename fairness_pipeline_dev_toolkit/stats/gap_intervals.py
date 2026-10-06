"""Confidence intervals and permutation tests for max−min gaps between groups.

A *gap* is ``max_k θ_k − min_k θ_k`` over K groups (demographic parity difference,
the TPR/FPR parts of equalized odds, MAE parity). Percentile bootstraps of a gap are
not calibrated when groups are equal: the statistic is non-negative, so the lower
bound is almost always above zero (BL-014).

The intervals here invert a *family-wise* set of pairwise-difference intervals
``[l_ij, u_ij]`` for ``θ_i − θ_j``. If the family covers every pairwise difference
simultaneously with probability ≥ ``level``, then the gap lies in
``[max_ij max(0, l_ij, −u_ij), max_ij max(|l_ij|, |u_ij|)]`` with probability
≥ ``level``. The guarantee is only as good as the pairwise intervals: Bonferroni
makes the family valid whenever each pairwise interval is valid, and the
pairwise intervals are large-sample approximations (Agresti–Caffo for rates,
Welch-t for means). The interval is conservative when groups differ a lot and
when K is large; it is wider than a percentile bootstrap.

All functions are deterministic except :func:`permutation_gap_pvalue`, which is
seeded by ``random_state``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from ..exceptions import IntervalUndefinedError

__all__ = [
    "simultaneous_gap_bounds",
    "binary_gap_interval",
    "equalized_odds_gap_interval",
    "welch_gap_interval",
    "permutation_gap_pvalue",
]

_TIE_TOL = 1e-12


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level!r}")


def _pairs(k: int) -> list[tuple[int, int]]:
    if k < 2:
        raise IntervalUndefinedError("too_few_groups", f"K={k} < 2")
    return list(combinations(range(k), 2))


def simultaneous_gap_bounds(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    max_gap: Optional[float] = 1.0,
) -> Tuple[float, float]:
    """Turn family-wise pairwise-difference intervals into an interval for the gap.

    Parameters
    ----------
    lower, upper
        Bounds ``l_ij <= u_ij`` of simultaneous intervals for every pairwise
        difference ``θ_i − θ_j`` (one entry per pair, any orientation).
    max_gap
        Upper limit of the gap's valid range (``1.0`` for rates, ``None`` for an
        unbounded statistic such as an MAE gap). Bounds are clipped to
        ``[0, max_gap]``.

    Returns
    -------
    (L, U)
        ``L = max_ij max(0, l_ij, −u_ij)`` and ``U = max_ij max(|l_ij|, |u_ij|)``.

    Guarantees
    ----------
    If the pairwise family covers all differences simultaneously with probability
    ≥ 1−α, then ``L <= gap <= U`` with probability ≥ 1−α. It does **not** make an
    invalid pairwise family valid, and it says nothing about *which* pair is extreme.

    Examples
    --------
    >>> simultaneous_gap_bounds([-0.05, 0.02], [0.05, 0.12])
    (0.02, 0.12)
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape or lo.size == 0:
        raise ValueError("lower and upper must be non-empty and the same shape")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise IntervalUndefinedError("nonfinite_pairwise_bounds")
    if np.any(lo > hi):
        raise ValueError("every lower bound must be <= its upper bound")
    big_l = float(np.max(np.maximum(0.0, np.maximum(lo, -hi))))
    big_u = float(np.max(np.maximum(np.abs(lo), np.abs(hi))))
    if max_gap is not None:
        big_l = min(big_l, max_gap)
        big_u = min(big_u, max_gap)
    return big_l, big_u


def _agresti_caffo_pairs(x: np.ndarray, n: np.ndarray, z: float) -> Tuple[np.ndarray, np.ndarray]:
    pt = (x + 1.0) / (n + 2.0)
    lo, hi = [], []
    for i, j in _pairs(x.size):
        d = pt[i] - pt[j]
        se = np.sqrt(pt[i] * (1 - pt[i]) / (n[i] + 2) + pt[j] * (1 - pt[j]) / (n[j] + 2))
        lo.append(max(d - z * se, -1.0))
        hi.append(min(d + z * se, 1.0))
    return np.asarray(lo), np.asarray(hi)


def _rate_gap(x: np.ndarray, n: np.ndarray) -> float:
    r = x / n
    return float(r.max() - r.min())


def _hull_with_point(bounds: Tuple[float, float], point: float) -> Tuple[float, float]:
    # Agresti–Caffo intervals are centred on shrunken rates, so at extreme gaps with
    # few rows (e.g. 0/10 vs 10/10 at level 0.8) U can fall below the observed gap.
    # Widening to include it can only raise coverage.
    return min(bounds[0], point), max(bounds[1], point)


def _validate_counts(successes: Sequence[float], n: Sequence[float]) -> Tuple[np.ndarray, ...]:
    x = np.asarray(successes, dtype=float)
    m = np.asarray(n, dtype=float)
    if x.shape != m.shape or x.ndim != 1:
        raise ValueError("successes and n must be 1-D arrays of the same length")
    if np.any(m <= 0):
        raise IntervalUndefinedError("empty_group", "every group needs at least one row")
    if np.any(x < 0) or np.any(x > m):
        raise ValueError("successes must satisfy 0 <= successes <= n")
    return x, m


def binary_gap_interval(
    successes: Sequence[float],
    n: Sequence[float],
    level: float = 0.95,
) -> Tuple[float, float]:
    """Simultaneous interval for the max−min gap of K binary rates (method M2a).

    Bonferroni Agresti–Caffo intervals for all ``K(K−1)/2`` pairwise rate
    differences, inverted by :func:`simultaneous_gap_bounds`. Analytic: no
    resampling, no randomness.

    Parameters
    ----------
    successes, n
        Positive counts and group sizes, one per group.
    level
        Family-wise confidence level.

    Guarantees
    ----------
    Coverage ≥ ``level`` up to the Agresti–Caffo large-sample approximation. The
    bounds always contain the observed gap (the interval is widened to it in the
    rare tiny-group, extreme-gap cases where the shrunken Agresti–Caffo centres
    would miss it). In the Wave 3a grid (K ∈ {2, 3, 5}, n_k from 30 to 3000, rates
    0.1/0.5, 500 datasets per cell) coverage at a true gap of 0 averaged 0.960
    with a minimum of 0.930. It is conservative away from equality and widens
    with K.

    Examples
    --------
    >>> lo, hi = binary_gap_interval([50, 50], [100, 100])
    >>> lo
    0.0
    """
    _check_level(level)
    x, m = _validate_counts(successes, n)
    pairs = _pairs(x.size)
    z = float(norm.ppf(1 - (1 - level) / (2 * len(pairs))))
    lo, hi = _agresti_caffo_pairs(x, m, z)
    return _hull_with_point(simultaneous_gap_bounds(lo, hi, max_gap=1.0), _rate_gap(x, m))


def equalized_odds_gap_interval(
    tp: Sequence[float],
    pos: Sequence[float],
    fp: Sequence[float],
    neg: Sequence[float],
    level: float = 0.95,
) -> Tuple[float, float]:
    """Simultaneous interval for ``max(TPR gap, FPR gap)`` over K groups.

    Agresti–Caffo intervals for every pairwise TPR difference and every pairwise
    FPR difference, with one Bonferroni correction over both families
    (``K(K−1)`` intervals). The equalized-odds gap is the max over both families,
    so :func:`simultaneous_gap_bounds` over the combined family bounds it.

    Parameters
    ----------
    tp, pos
        True positives and actual positives per group.
    fp, neg
        False positives and actual negatives per group.

    Guarantees
    ----------
    Same as :func:`binary_gap_interval`, including containment of the observed
    gap. In the Wave 3a grid (prevalence 0.4, FPR/TPR 0.1/0.9 and 0.5/0.5,
    4000 datasets per cell at a true gap of 0) coverage averaged 0.962 with a
    minimum of 0.932 (K=2, sizes 30 and 3000, rates 0.5).

    Raises
    ------
    IntervalUndefinedError
        ``reason="empty_label_stratum"`` if any group has no positives or no
        negatives (its TPR or FPR is undefined).

    Examples
    --------
    >>> lo, hi = equalized_odds_gap_interval([40, 40], [50, 50], [10, 10], [50, 50])
    >>> lo
    0.0
    """
    _check_level(level)
    tp_a, pos_a = np.asarray(tp, float), np.asarray(pos, float)
    fp_a, neg_a = np.asarray(fp, float), np.asarray(neg, float)
    if np.any(pos_a <= 0) or np.any(neg_a <= 0):
        raise IntervalUndefinedError(
            "empty_label_stratum", "every group needs at least one positive and one negative"
        )
    t_x, t_n = _validate_counts(tp_a, pos_a)
    f_x, f_n = _validate_counts(fp_a, neg_a)
    n_pairs = len(_pairs(t_x.size))
    z = float(norm.ppf(1 - (1 - level) / (2 * 2 * n_pairs)))
    lo_t, hi_t = _agresti_caffo_pairs(t_x, t_n, z)
    lo_f, hi_f = _agresti_caffo_pairs(f_x, f_n, z)
    bounds = simultaneous_gap_bounds(
        np.concatenate([lo_t, lo_f]), np.concatenate([hi_t, hi_f]), max_gap=1.0
    )
    return _hull_with_point(bounds, max(_rate_gap(t_x, t_n), _rate_gap(f_x, f_n)))


def welch_gap_interval(
    means: Sequence[float],
    variances: Sequence[float],
    n: Sequence[float],
    level: float = 0.95,
) -> Tuple[float, float]:
    """Simultaneous interval for the max−min gap of K group means (Bonferroni Welch-t).

    Each pairwise difference gets a Welch-t interval at level ``1 − α/P`` with the
    Welch–Satterthwaite degrees of freedom; the family is inverted by
    :func:`simultaneous_gap_bounds` with no upper clip.

    Guarantees
    ----------
    Only as good as the t approximation for each group mean. It is **not**
    calibrated for skewed data in small groups: for lognormal(σ=1) absolute
    errors with a 30-row group beside a 3000-row group, coverage of a zero gap was
    0.88 at ``level=0.95``. For that reason the analyzer does not use it as the
    MAE-gap default (``mae_gap`` reports an undefined CI); it is exposed for users
    who have checked the approximation for their data.

    Parameters
    ----------
    means, variances, n
        Per-group sample means, sample variances (``ddof=1``) and sizes.

    Raises
    ------
    IntervalUndefinedError
        ``"group_too_small"`` if any group has fewer than 2 rows, or
        ``"zero_variance"`` if a pair has zero estimated standard error (a
        zero-width interval would be overconfident).

    Examples
    --------
    >>> lo, hi = welch_gap_interval([1.0, 1.0], [1.0, 1.0], [100, 100])
    >>> lo
    0.0
    """
    _check_level(level)
    mu = np.asarray(means, float)
    var = np.asarray(variances, float)
    m = np.asarray(n, float)
    if not (mu.shape == var.shape == m.shape) or mu.ndim != 1:
        raise ValueError("means, variances and n must be 1-D arrays of the same length")
    if np.any(m < 2):
        raise IntervalUndefinedError("group_too_small", "Welch-t needs at least 2 rows per group")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(var))):
        raise IntervalUndefinedError("nonfinite_group_statistics")
    pairs = _pairs(mu.size)
    a_tail = (1 - level) / (2 * len(pairs))
    lo, hi = [], []
    for i, j in pairs:
        a, b = var[i] / m[i], var[j] / m[j]
        se = np.sqrt(a + b)
        if se <= 0:
            raise IntervalUndefinedError("zero_variance", "a pair of groups has zero spread")
        df = (a + b) ** 2 / (a**2 / (m[i] - 1) + b**2 / (m[j] - 1))
        crit = float(student_t.ppf(1 - a_tail, df))
        d = mu[i] - mu[j]
        lo.append(d - crit * se)
        hi.append(d + crit * se)
    return simultaneous_gap_bounds(lo, hi, max_gap=None)


def _binary_permutation_pvalue(
    blocks: list,
    k: int,
    observed: float,
    tol: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Label shuffles for 0/1 values, drawn at the count level.

    Shuffling group labels over 0/1 values only changes how each stratum's
    positives split across groups of fixed sizes, which is a sequential
    hypergeometric draw. Same null distribution as row shuffles, O(K) per draw.
    """
    stat = np.zeros(n_permutations)
    for _codes, vals, counts in blocks:
        good = np.full(n_permutations, int(vals.sum()), dtype=np.int64)
        bad = np.full(n_permutations, int(vals.size - vals.sum()), dtype=np.int64)
        sums = np.empty((n_permutations, k))
        for j in range(k - 1):
            x = rng.hypergeometric(good, bad, int(counts[j])) if counts[j] > 0 else 0
            sums[:, j] = x
            good -= x
            bad -= int(counts[j]) - x
        sums[:, k - 1] = good
        stat = np.maximum(stat, _stratum_gap(sums, counts))
    return (1 + int(np.sum(stat >= observed - tol))) / (n_permutations + 1)


def _stratum_gap(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    means = sums / counts
    return means.max(axis=-1) - means.min(axis=-1)


def permutation_gap_pvalue(
    values: Sequence[float],
    groups: Sequence[int],
    *,
    strata: Optional[Sequence[int]] = None,
    n_permutations: int = 2000,
    random_state: Optional[int] = 42,
) -> float:
    """Permutation p-value for H0 "every group has the same mean" using the gap statistic.

    The statistic is the max−min gap of group means of ``values``. With
    ``strata``, group labels are shuffled *within* each stratum, the gap is
    computed per stratum, and the statistic is the max over strata (for
    equalized odds: ``values=y_pred``, ``strata=y_true`` gives
    ``max(TPR gap, FPR gap)``).

    Parameters
    ----------
    values
        Per-row values (``y_pred`` for DPD, absolute errors for MAE).
    groups
        Integer group codes ``0..K-1`` per row. Rows outside the analysis should
        be removed before calling.
    strata
        Optional integer stratum per row.
    n_permutations
        Number of label shuffles (default 2000).
    random_state
        Seed; the p-value is reproducible for a fixed seed.

    Returns
    -------
    float
        ``(1 + #{T* >= T_obs}) / (n_permutations + 1)``; exact under
        exchangeability of group labels (within strata).

    Notes
    -----
    For 0/1 ``values`` the shuffles are drawn at the count level (sequential
    hypergeometric split of each stratum's positives), which has the same null
    distribution as row shuffles but costs O(K) per draw instead of O(N).

    Raises
    ------
    IntervalUndefinedError
        ``"empty_label_stratum"`` if some group has no rows in some stratum.

    Examples
    --------
    >>> p = permutation_gap_pvalue([0, 1, 0, 1], [0, 0, 1, 1], n_permutations=99)
    >>> 0 < p <= 1
    True
    """
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=np.int64)
    s = np.zeros_like(g) if strata is None else np.asarray(strata, dtype=np.int64)
    if not (v.shape == g.shape == s.shape) or v.ndim != 1:
        raise ValueError("values, groups and strata must be 1-D and the same length")
    k = int(g.max()) + 1 if g.size else 0
    _pairs(k)
    blocks = []
    for level_s in np.unique(s):
        idx = np.flatnonzero(s == level_s)
        counts = np.bincount(g[idx], minlength=k).astype(float)
        if np.any(counts == 0):
            raise IntervalUndefinedError(
                "empty_label_stratum", "a group has no rows in a label stratum"
            )
        blocks.append((g[idx], v[idx], counts))

    def stat(code_rows: list[np.ndarray]) -> np.ndarray:
        out = None
        for (codes0, vals, counts), codes in zip(blocks, code_rows):
            c = codes.shape[0]
            flat = (np.arange(c)[:, None] * k + codes).ravel()
            sums = np.bincount(flat, weights=np.tile(vals, c), minlength=c * k).reshape(c, k)
            gap = _stratum_gap(sums, counts)
            out = gap if out is None else np.maximum(out, gap)
        return out

    observed = float(stat([codes0[None, :] for codes0, _, _ in blocks])[0])
    tol = _TIE_TOL * max(1.0, abs(observed))
    rng = np.random.default_rng(random_state)
    if np.all((v == 0.0) | (v == 1.0)):
        return _binary_permutation_pvalue(blocks, k, observed, tol, n_permutations, rng)
    n_rows = max(v.size, 1)
    chunk = max(1, min(n_permutations, 4_000_000 // n_rows))
    exceed = 0
    done = 0
    while done < n_permutations:
        c = min(chunk, n_permutations - done)
        shuffled = [
            rng.permuted(np.broadcast_to(codes0, (c, codes0.size)), axis=1)
            for codes0, _, _ in blocks
        ]
        exceed += int(np.sum(stat(shuffled) >= observed - tol))
        done += c
    return (1 + exceed) / (n_permutations + 1)
