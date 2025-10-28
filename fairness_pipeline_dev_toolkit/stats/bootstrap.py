from __future__ import annotations
from typing import Callable, Tuple, Optional
import numpy as np

def _percentile_ci(samples: np.ndarray, level: float) -> Tuple[float, float]:
    alpha = (1 - level) / 2
    lower = np.percentile(samples, alpha)
    upper = np.percentile(samples, 1 - alpha)
    return float(lower), float(upper)


# def bootstrap_ci(values, stat_fn: Callable, B: int = 2000, level: float = 0.95):
#     values = np.asarray(values)
#     if len(values) == 0:
#         return (np.nan, np.nan)
#     stats = []
#     n = len(values)
#     rng = np.random.default_rng(42)
#     for _ in range(B):
#         sample = values[rng.integers(0, n, n)]
#         stats.append(stat_fn(sample))
#     lower = np.percentile(stats, (1 - level) / 2 * 100)
#     upper = np.percentile(stats, (1 + level) / 2 * 100)
#     return float(lower), float(upper)

def bootstrap_ci(
        data: np.ndarray,
        stat_fn: Callable[[np.ndarray], float],
        *,
        B: int = 2000,
        level: float = 0.95,
        method: str = "percentile",
        random_state: Optional[int] = 42
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
    """
    x = np.asarray(data)
    n = x.shape[0]
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    stats = np.empty(B, dtype=float)
    for b in range(B):
        sample = x[rng.integers(0, n, n)]
        stats[b] = stat_fn(sample)

    if method == "percentile":
        return _percentile_ci(stats, level)
    elif method == "bca":
        return bca_ci(x, stat_fn, stats, level)
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")
    
def bca_ci(
        x: np.ndarray,
        stat_fn: Callable[[np.ndarray], float],
        boot_stats: np.ndarray,
        *,
        level: float = 0.95
    ) -> Tuple[float, float]:
    """
    Bias-Corrected and Accelerated (BCa) bootstrap confidence interval.

    Based on Efron and Tibshirani (1993). Comutes bias correction (z0) via proportion of
    bootstrap statistics less than the observed statistic, and acceleration (a) via jackknife.

    Notes: 
    - BCa intervals can be unstable for small samples or extreme statistics; fallback if needed.
    Args:
        x: Original data array.
        stat_fn: Statistic function.
        boot_stats: Precomputed bootstrap statistics.
        level: Confidence level for the interval.
    Returns:
        A tuple (lower_bound, upper_bound) representing the BCa confidence interval.
    """
    x = np.asarray(x)
    n = x.shape[0]
    if n < 5 or np.any(~np.isfinite(x)):
        # BCa unreliable for small samples or non-finite data
        return _percentile_ci(boot_stats, level)
    
    theta_hat = stat_fn(x)
    # bias correction z0
    prop_less = np.mean(boot_stats < theta_hat)
    # guard for p==0 or p==1
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = _z(prop_less)

    # acceleration a via jackknife
    jack_stats = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = stat_fn(np.delete(x, i))
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
    a = num / denom if denom != 0 else 0.0

    alpha1 = (1 - level) / 2
    alpha2 = 1 - alpha1

    z_alpha1 = _z(alpha1)
    z_alpha2 = _z(alpha2)

    def bca_quantile(alpha):
        z = _z(alpha)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        return _phi(adj)
    
    q1 = bca_quantile(alpha1)
    q2 = bca_quantile(alpha2)
    lower = np.percentile(boot_stats, q1)
    upper = np.percentile(boot_stats, q2)
    return float(lower), float(upper)


#------------------helpers------------------#
def _z(p: float) -> float:
    """Inverse of standard normal CDF (probit function)."""
    # Use numpy's erfcinv-based approximation
    # z = sqrt(2) * erfinv(2p) * -1
    from math import sqrt
    from numpy import erfcinv
    return sqrt(2) * float(erfcinv(2 *  p))

def _phi(z: float) -> float:
    """Standard normal CDF."""
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / np.sqrt(2)))

