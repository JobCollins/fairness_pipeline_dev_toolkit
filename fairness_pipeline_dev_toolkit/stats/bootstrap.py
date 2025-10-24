from typing import Callable, Tuple
import numpy as np


def bootstrap_ci(values, stat_fn: Callable, B: int = 2000, level: float = 0.95):
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan)
    stats = []
    n = len(values)
    rng = np.random.default_rng(42)
    for _ in range(B):
        sample = values[rng.integers(0, n, n)]
        stats.append(stat_fn(sample))
    lower = np.percentile(stats, (1 - level) / 2 * 100)
    upper = np.percentile(stats, (1 + level) / 2 * 100)
    return float(lower), float(upper)