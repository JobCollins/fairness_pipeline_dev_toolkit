from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import beta as beta_dist


def beta_binomial_interval(
    successes: int, trials: int, *, level: float = 0.95, alpha: float = 1.0, beta: float = 1.0
) -> Tuple[float, float]:
    """
    Bayesian credible interval for a Bernoulli proportion using a Beta prior.
    Use weakly informative Beta(1,1) prior by default.
    Args:
        successes: Number of successful outcomes.
        trials: Total number of trials.
        level: Credible interval level. Default is 0.95.
        alpha: Alpha parameter of the Beta prior. Default is 1.0.
        beta: Beta parameter of the Beta prior. Default is 1.0.
    Returns:
        A tuple (lower_bound, upper_bound) such that P(p in [lower_bound, upper_bound] | data) = level.
    """
    if trials <= 0:
        return (np.nan, np.nan)
    a_post = successes + alpha
    b_post = (trials - successes) + beta
    lower = beta_dist.ppf((1 - level) / 2, a_post, b_post)
    upper = beta_dist.ppf(1 - (1 - level) / 2, a_post, b_post)
    return float(lower), float(upper)
