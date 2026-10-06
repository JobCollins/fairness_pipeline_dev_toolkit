from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import BootstrapUndefinedError


def _require_finite_replicates(samples: np.ndarray) -> np.ndarray:
    boot = np.asarray(samples, dtype=float)
    bad = int(np.sum(~np.isfinite(boot)))
    if boot.size == 0:
        raise BootstrapUndefinedError("no_replicates")
    if bad:
        raise BootstrapUndefinedError(
            "nonfinite_replicates", f"{bad} of {boot.size} bootstrap replicates are not finite"
        )
    return boot


def _percentile_ci(samples: np.ndarray, level: float) -> Tuple[float, float]:
    """Percentile CI. Refuses (``BootstrapUndefinedError``) if any replicate is non-finite.

    Dropping non-finite replicates would condition the interval on resamples that
    happened to keep every group, which is a different estimand.
    """
    boot = _require_finite_replicates(samples)
    alpha = (1 - level) / 2
    lower = np.percentile(boot, alpha * 100)
    upper = np.percentile(boot, (1 - alpha) * 100)
    return float(lower), float(upper)


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    B: int = 2000,
    level: float = 0.95,
    method: str = "percentile",
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """
    Generic bootstrap confidence interval computation on 1D data vectors.

    Args:
        data: 1D vector of values (e.g., per-row contributions, group rates).
        stat_fn: Function that maps a 1D array -> float (statistic to bootstrap).
        B: Number of bootstrap samples to draw. Default is 2000.
        level: Confidence level for the interval. Default is 0.95.
        method: {"percentile","bca"}
        Percentile = robust default; BCa = bias-corrected & accelerated. Method for computing the confidence interval.
        random_state: Seed for the random number generator for reproducibility. Default is 42.

    Returns:
        A tuple (lower_bound, upper_bound) representing the confidence interval.

    Raises:
        BootstrapUndefinedError: if ``data`` is empty, or any replicate (or, for BCa,
            the estimate, jackknife or bias correction) is non-finite. A NaN interval
            is never returned.

    Not for max−min gaps between groups: percentile and BCa intervals of a gap are
    not calibrated at equality (BL-014). Use :mod:`fairness_pipeline_dev_toolkit.stats.gap_intervals`.
    """
    if method not in ("percentile", "bca"):
        raise ValueError(f"Unknown bootstrap method: {method}")
    x = np.asarray(data)
    n = x.shape[0]
    if n == 0:
        raise BootstrapUndefinedError("empty_data")
    rng = np.random.default_rng(random_state)
    stats = np.empty(B, dtype=float)
    for b in range(B):
        sample = x[rng.integers(0, n, n)]
        stats[b] = stat_fn(sample)

    if method == "percentile":
        return _percentile_ci(stats, level)
    return bca_ci(x, stat_fn, stats, level=level)


def stratified_bootstrap_replicates(
    strata: Sequence[np.ndarray],
    stat_fn: Callable[[np.ndarray], float],
    *,
    B: int = 2000,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Bootstrap replicates resampling *within* each stratum.

    Each replicate draws ``len(s)`` indices with replacement from every stratum ``s``
    and passes the concatenated indices to ``stat_fn``. Every stratum keeps its size,
    so no replicate can lose a group (the source of NaN replicates in an
    unstratified index bootstrap).

    Args:
        strata: Index arrays, one per stratum (e.g. per group, or per group × y_true).
        stat_fn: Maps an index array to a float.
        B: Number of replicates.
        random_state: Seed.

    Returns:
        Array of ``B`` replicate statistics.
    """
    parts = [np.asarray(s, dtype=int) for s in strata]
    if not parts or any(p.size == 0 for p in parts):
        raise BootstrapUndefinedError("empty_stratum", "every stratum needs at least one row")
    rng = np.random.default_rng(random_state)
    stats = np.empty(B, dtype=float)
    for b in range(B):
        idx = np.concatenate([p[rng.integers(0, p.size, p.size)] for p in parts])
        stats[b] = stat_fn(idx)
    return stats


def bootstrap_difference_of_means(
    a: np.ndarray,
    b: np.ndarray,
    *,
    B: int = 2000,
    level: float = 0.95,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Percentile CI for ``mean(a) - mean(b)`` with independent resamples of each sample.

    Raises ``BootstrapUndefinedError`` if either sample is empty.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n_a, n_b = x.shape[0], y.shape[0]
    if n_a == 0 or n_b == 0:
        raise BootstrapUndefinedError("empty_data")
    rng = np.random.default_rng(random_state)
    stats = np.empty(B, dtype=float)
    for i in range(B):
        sample_a = x[rng.integers(0, n_a, n_a)]
        sample_b = y[rng.integers(0, n_b, n_b)]
        stats[i] = float(np.mean(sample_a) - np.mean(sample_b))
    return _percentile_ci(stats, level)


def bca_ci(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    boot_stats: np.ndarray,
    *,
    level: float = 0.95,
) -> Tuple[float, float]:
    """
    Bias-Corrected and Accelerated (BCa) bootstrap confidence interval.

    Based on Efron and Tibshirani (1993). Computes bias correction (z0) via proportion of
    bootstrap statistics less than the observed statistic, and acceleration (a) via jackknife.

    Refuse policy (BL-031): raises ``BootstrapUndefinedError`` instead of returning a
    NaN interval, falling back to percentile, or dropping replicates when

    - any bootstrap replicate is non-finite (``reason="nonfinite_replicates"``),
    - the estimate ``stat_fn(x)`` is non-finite (``"nonfinite_estimate"``),
    - any jackknife value, or the acceleration, is non-finite (``"nonfinite_jackknife"``),
    - the bias correction ``z0`` or the adjusted quantiles are non-finite
      (``"nonfinite_z0"``).

    Args:
        x: Original data array (or analysis-row indices, if ``stat_fn`` takes indices).
        stat_fn: Statistic function.
        boot_stats: Precomputed bootstrap statistics.
        level: Confidence level for the interval.
    Returns:
        A tuple (lower_bound, upper_bound) representing the BCa confidence interval.
    """
    x = np.asarray(x)
    n = x.shape[0]
    if n == 0:
        raise BootstrapUndefinedError("empty_data")
    boot = _require_finite_replicates(boot_stats)

    theta_hat = float(stat_fn(x))
    if not np.isfinite(theta_hat):
        raise BootstrapUndefinedError("nonfinite_estimate")
    # p == 0 or p == 1 means z0 = ∓inf: the estimate lies outside every replicate.
    prop_less = float(np.mean(boot < theta_hat))
    z0 = _z(prop_less) if 0.0 < prop_less < 1.0 else float("nan")
    if not np.isfinite(z0):
        raise BootstrapUndefinedError(
            "nonfinite_z0", f"share of replicates below the estimate is {prop_less:g}"
        )

    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = stat_fn(np.delete(x, i))
    n_bad = int(np.sum(~np.isfinite(jack)))
    if n_bad:
        raise BootstrapUndefinedError(
            "nonfinite_jackknife", f"{n_bad} of {n} leave-one-out values are not finite"
        )
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
    a = num / denom
    if not np.isfinite(a):
        raise BootstrapUndefinedError("nonfinite_jackknife", "acceleration is not finite")

    alpha1 = (1 - level) / 2
    alpha2 = 1 - alpha1

    def bca_quantile(alpha):
        z = _z(alpha)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        return _phi(adj)

    q1 = bca_quantile(alpha1)
    q2 = bca_quantile(alpha2)
    if not (np.isfinite(q1) and np.isfinite(q2)):
        raise BootstrapUndefinedError("nonfinite_z0", "adjusted BCa quantiles are not finite")
    # bca_quantile returns probabilities in (0, 1); np.percentile expects percent in [0, 100].
    lower = np.percentile(boot, q1 * 100)
    upper = np.percentile(boot, q2 * 100)
    return float(lower), float(upper)


# ------------------helpers------------------#
def _z(p: float) -> float:
    """Inverse of standard normal CDF (probit function)."""
    # Use scipy's norm.ppf which is the inverse of the standard normal CDF
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:
        # Fallback: use erfcinv if scipy.stats not available but scipy.special is
        from math import sqrt

        from scipy.special import erfcinv

        # erfcinv(2p) gives the value where erfc(x) = 2p
        # For probit: z = sqrt(2) * erfinv(2p - 1)
        # Using erfcinv: z = sqrt(2) * erfcinv(2 - 2p) with sign adjustment
        return sqrt(2) * float(erfcinv(2 * (1 - p)))


def _phi(z: float) -> float:
    """Standard normal CDF."""
    from math import erf

    return 0.5 * (1 + erf(z / np.sqrt(2)))
