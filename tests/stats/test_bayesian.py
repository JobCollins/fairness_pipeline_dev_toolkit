"""
Unit tests for Bayesian statistical functions.
"""

from __future__ import annotations

import numpy as np

from fairness_pipeline_dev_toolkit.stats.bayesian import beta_binomial_interval


class TestBetaBinomialInterval:
    """Tests for beta_binomial_interval function."""

    def test_basic_beta_binomial_interval(self):
        """Test basic beta-binomial interval calculation."""
        lower, upper = beta_binomial_interval(successes=10, trials=20, level=0.95)

        assert lower < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_beta_binomial_interval_all_successes(self):
        """Test with all successes."""
        lower, upper = beta_binomial_interval(successes=10, trials=10, level=0.95)

        assert lower < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        # Upper bound should be high (close to 1)
        assert upper > 0.5

    def test_beta_binomial_interval_no_successes(self):
        """Test with no successes."""
        lower, upper = beta_binomial_interval(successes=0, trials=10, level=0.95)

        assert lower < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        # Lower bound should be low (close to 0)
        assert lower < 0.5

    def test_beta_binomial_interval_small_n(self):
        """Test with small sample size."""
        lower, upper = beta_binomial_interval(successes=2, trials=5, level=0.95)

        assert lower < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_beta_binomial_interval_custom_prior(self):
        """Test with custom Beta prior parameters."""
        lower1, upper1 = beta_binomial_interval(
            successes=10, trials=20, level=0.95, alpha=1.0, beta=1.0
        )
        lower2, upper2 = beta_binomial_interval(
            successes=10, trials=20, level=0.95, alpha=2.0, beta=2.0
        )

        # Different priors should give different intervals
        assert (lower1, upper1) != (lower2, upper2)

    def test_beta_binomial_interval_different_levels(self):
        """Test with different confidence levels."""
        lower_90, upper_90 = beta_binomial_interval(successes=10, trials=20, level=0.90)
        lower_95, upper_95 = beta_binomial_interval(successes=10, trials=20, level=0.95)
        lower_99, upper_99 = beta_binomial_interval(successes=10, trials=20, level=0.99)

        # Higher level should give wider intervals
        assert (upper_99 - lower_99) >= (upper_95 - lower_95)
        assert (upper_95 - lower_95) >= (upper_90 - lower_90)

    def test_beta_binomial_interval_zero_trials_returns_nan(self):
        """Test that zero trials returns NaN."""
        lower, upper = beta_binomial_interval(successes=0, trials=0, level=0.95)

        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_beta_binomial_interval_negative_trials_returns_nan(self):
        """Test that negative trials returns NaN."""
        lower, upper = beta_binomial_interval(successes=5, trials=-1, level=0.95)

        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_beta_binomial_interval_successes_exceed_trials(self):
        """Test with successes exceeding trials (edge case)."""
        # This shouldn't happen in practice, but test robustness
        lower, upper = beta_binomial_interval(successes=15, trials=10, level=0.95)

        # Should still compute (posterior will handle it)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_beta_binomial_interval_balanced_case(self):
        """Test with balanced case (50% success rate)."""
        lower, upper = beta_binomial_interval(successes=50, trials=100, level=0.95)

        # Interval should be roughly centered around 0.5
        assert lower < 0.5 < upper
        assert abs((lower + upper) / 2 - 0.5) < 0.2

    def test_beta_binomial_interval_monotonic_with_successes(self):
        """Test that intervals are monotonic as successes increase."""
        # From test_stats_bootstrap.py - small-n posterior intervals shrink as successes approach trials
        low = beta_binomial_interval(successes=1, trials=5, level=0.95)
        mid = beta_binomial_interval(successes=3, trials=5, level=0.95)
        high = beta_binomial_interval(successes=5, trials=5, level=0.95)

        assert low[1] < high[1] and low[0] < mid[0] < high[0]

    def test_beta_binomial_interval_weakly_informative_prior(self):
        """Test with weakly informative prior (default Beta(1,1))."""
        lower1, upper1 = beta_binomial_interval(successes=10, trials=20, level=0.95)
        lower2, upper2 = beta_binomial_interval(
            successes=10, trials=20, level=0.95, alpha=1.0, beta=1.0
        )

        # Should be the same (default is Beta(1,1))
        assert abs(lower1 - lower2) < 1e-10
        assert abs(upper1 - upper2) < 1e-10

    def test_beta_binomial_interval_informative_prior(self):
        """Test with informative prior."""
        # Strong prior favoring low probability
        lower, upper = beta_binomial_interval(
            successes=5, trials=10, level=0.95, alpha=1.0, beta=9.0
        )

        # Prior pulls estimate down, so interval should be lower
        assert upper < 0.7  # Should be pulled down by prior

    def test_beta_binomial_interval_large_n(self):
        """Test with large sample size."""
        lower, upper = beta_binomial_interval(successes=500, trials=1000, level=0.95)

        # With large n, interval should be narrow
        assert (upper - lower) < 0.1
        # And centered around 0.5
        assert abs((lower + upper) / 2 - 0.5) < 0.05
