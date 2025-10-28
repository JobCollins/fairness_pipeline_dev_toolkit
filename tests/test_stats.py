import numpy as np
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci
from fairness_pipeline_dev_toolkit.stats.bayesian import beta_binomial_interval

def two_sample_percentile_ci(A, B, level=0.90, B_reps=1000, rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    nA, nB = len(A), len(B)
    stats = np.empty(B_reps, dtype=float)
    for b in range(B_reps):
        A_star = A[rng.integers(0, nA, size=nA)]
        B_star = B[rng.integers(0, nB, size=nB)]
        stats[b] = A_star.mean() - B_star.mean()
    alpha = 1 - level
    lo = np.percentile(stats, 100 * (alpha / 2))
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def test_bootstrap_percentile_coverage_simple():
    # Two groups with true rates pA=0.6, pB=0.5; true disparity = 0.1
    rng = np.random.default_rng(123)
    level = 0.90
    trials = 400
    sims = 200
    covered = 0

    for s in range(sims):
        A = rng.binomial(1, 0.6, trials)
        B = rng.binomial(1, 0.5, trials)
        ci = two_sample_percentile_ci(A, B, level=level, B_reps=1000, rng_seed=1000 + s)
        if ci[0] <= 0.1 <= ci[1]:
            covered += 1
    est_coverage = covered / sims
    # Allow some slack due to Monte Carlo error
    assert 0.85 <= est_coverage <= 0.95

def test_beta_binomial_interval_small_n_monotonic():
    # Small-n posterior intervals shrink as successes approach trials
    low = beta_binomial_interval(successes=1, trials=5, level=0.95)
    mid = beta_binomial_interval(successes=3, trials=5, level=0.95)
    high = beta_binomial_interval(successes=5, trials=5, level=0.95)
    assert low[1] < high[1] and low[0] < mid[0] < high[0]
