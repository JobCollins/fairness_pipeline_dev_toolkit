from typing import Tuple
import numpy as np


def beta_binomial_interval(successes: int, trials: int, level: float = 0.95, alpha: float = 1.0, beta: float = 1.0) -> Tuple[float, float]:
    if trials == 0:
        return (np.nan, np.nan)
    from scipy.stats import beta as beta_dist
    lower = beta_dist.ppf((1 - level) / 2, successes + alpha, trials - successes + beta)
    upper = beta_dist.ppf((1 + level) / 2, successes + alpha, trials - successes + beta)
    return float(lower), float(upper)