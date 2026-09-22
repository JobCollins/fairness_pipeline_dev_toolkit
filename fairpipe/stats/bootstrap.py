from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    stat_fn,
    B: int = 1000,
    level: float = 0.95,
    method: str = "percentile",
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Data array (or array of indices) to bootstrap.
    stat_fn : callable
        Function taking a resampled data/index array and returning a float.
    B : int
        Number of bootstrap samples.
    level : float
        Confidence level (e.g. 0.95).
    method : str
        Method for confidence interval: "percentile" or "bca".

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound)
    """
    data = np.asarray(data)
    n = data.shape[0]
    if n == 0:
        return (np.nan, np.nan)

    theta_hat = stat_fn(data)
    if not np.isfinite(theta_hat):
        return (np.nan, np.nan)

    boot_stats = np.empty(B)
    rng = np.random.default_rng()
    for i in range(B):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    boot_stats = boot_stats[np.isfinite(boot_stats)]
    if boot_stats.size == 0:
        return (np.nan, np.nan)

    alpha = 1.0 - level

    if method == "percentile":
        low_pct = (alpha / 2.0) * 100.0
        high_pct = (1.0 - alpha / 2.0) * 100.0
        return (
            float(np.percentile(boot_stats, low_pct)),
            float(np.percentile(boot_stats, high_pct)),
        )

    elif method == "bca":
        p_less = np.mean(boot_stats < theta_hat)
        p_eq = np.mean(boot_stats == theta_hat)
        prop = p_less + 0.5 * p_eq
        prop = np.clip(prop, 1e-10, 1.0 - 1e-10)
        z0 = stats.norm.ppf(prop)

        jack_stats = np.empty(n)
        for i in range(n):
            loo_sample = np.concatenate([data[:i], data[i + 1 :]])
            jack_stats[i] = stat_fn(loo_sample)

        jack_mean = np.mean(jack_stats)
        num = np.sum((jack_mean - jack_stats) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)

        a = 0.0 if den == 0.0 or not np.isfinite(den) else num / den
        if not np.isfinite(a):
            a = 0.0

        z_alpha = stats.norm.ppf(alpha / 2.0)
        z_1_alpha = stats.norm.ppf(1.0 - alpha / 2.0)

        def compute_pct(z_val):
            denom = 1.0 - a * (z0 + z_val)
            if denom == 0.0 or not np.isfinite(denom):
                return np.nan
            val = z0 + (z0 + z_val) / denom
            return stats.norm.cdf(val) * 100.0

        p1 = compute_pct(z_alpha)
        p2 = compute_pct(z_1_alpha)

        if not np.isfinite(p1) or not np.isfinite(p2):
            low_pct = (alpha / 2.0) * 100.0
            high_pct = (1.0 - alpha / 2.0) * 100.0
            return (
                float(np.percentile(boot_stats, low_pct)),
                float(np.percentile(boot_stats, high_pct)),
            )

        p1 = np.clip(p1, 0.0, 100.0)
        p2 = np.clip(p2, 0.0, 100.0)

        return (
            float(np.percentile(boot_stats, p1)),
            float(np.percentile(boot_stats, p2)),
        )
    else:
        raise ValueError(f"Unknown bootstrap CI method: {method}")