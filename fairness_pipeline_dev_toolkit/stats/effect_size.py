from __future__ import annotations

import numpy as np


def risk_ratio(p1: float, p2: float) -> float:
    """
    Risk (rate) ratio = p1 / p2.
    Args:
        p1: Rate for group 1 (numerator).
        p2: Rate for group 2 (denominator).
    Returns:
        Risk ratio as a float.
        Returns np.nan if denominator is zero or any input is invalid.
    """
    if p1 is None or p2 is None:
        return np.nan
    if not np.isfinite(p1) or not np.isfinite(p2) or p2 == 0:
        return np.nan
    return float(p1 / p2)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d effect size for two samples: (mean_x - mean_y)/pooled_std.
    Args:
        x: 1D array of values for group 1.
        y: 1D array of values for group 2.
    Returns:
        Cohen's d as a float.
        Returns np.nan if inputs are invalid or have insufficient data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    # pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return np.nan
    return float((mean_x - mean_y) / pooled_std)
