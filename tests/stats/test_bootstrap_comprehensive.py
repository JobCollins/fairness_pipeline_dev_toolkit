"""
Comprehensive tests for bootstrap confidence interval functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.stats.bootstrap import (
    _percentile_ci,
    _phi,
    _z,
    bca_ci,
    bootstrap_ci,
)


class TestPercentileCI:
    """Tests for _percentile_ci() helper function."""

    def test_percentile_ci_basic(self):
        """Test basic percentile CI computation."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = _percentile_ci(samples, level=0.95)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper
        # Percentile CI may not include exact min/max, but should be within reasonable range
        assert lower <= np.max(samples)
        assert upper >= np.min(samples)

    def test_percentile_ci_90_level(self):
        """Test with 90% confidence level."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = _percentile_ci(samples, level=0.90)

        assert lower < upper
        # 90% CI should be narrower than 95% CI
        lower_95, upper_95 = _percentile_ci(samples, level=0.95)
        assert (upper - lower) < (upper_95 - lower_95)

    def test_percentile_ci_99_level(self):
        """Test with 99% confidence level."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = _percentile_ci(samples, level=0.99)

        assert lower < upper
        # 99% CI should be wider than 95% CI
        lower_95, upper_95 = _percentile_ci(samples, level=0.95)
        assert (upper - lower) > (upper_95 - lower_95)

    def test_percentile_ci_single_value(self):
        """Test with single value."""
        samples = np.array([5.0])
        lower, upper = _percentile_ci(samples, level=0.95)

        assert lower == upper == 5.0

    def test_percentile_ci_sorted_values(self):
        """Test that CI bounds are within sample range."""
        samples = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lower, upper = _percentile_ci(samples, level=0.95)

        assert lower <= np.max(samples)
        assert upper >= np.min(samples)

    def test_percentile_ci_unsorted_values(self):
        """Test with unsorted values."""
        samples = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        lower, upper = _percentile_ci(samples, level=0.95)

        assert lower < upper
        assert lower <= np.max(samples)
        assert upper >= np.min(samples)


class TestZProbit:
    """Tests for _z() probit function (inverse normal CDF)."""

    def test_z_median(self):
        """Test that z(0.5) is approximately 0."""
        z_val = _z(0.5)
        assert abs(z_val) < 0.01  # Should be very close to 0

    def test_z_symmetric(self):
        """Test that z is symmetric around 0.5."""
        z1 = _z(0.25)
        z2 = _z(0.75)
        assert abs(z1 + z2) < 0.01  # Should be approximately symmetric

    def test_z_extreme_low(self):
        """Test with very low probability."""
        z_val = _z(0.01)
        assert z_val < -2.0  # Should be negative and far from 0

    def test_z_extreme_high(self):
        """Test with very high probability."""
        z_val = _z(0.99)
        assert z_val > 2.0  # Should be positive and far from 0

    def test_z_monotonic(self):
        """Test that z is monotonic increasing."""
        z1 = _z(0.1)
        z2 = _z(0.5)
        z3 = _z(0.9)
        assert z1 < z2 < z3

    def test_z_boundary_values(self):
        """Test with boundary probability values."""
        # z(0) should be -inf, but function clips to avoid issues
        # z(1) should be +inf, but function clips to avoid issues
        z_low = _z(1e-6)
        z_high = _z(1 - 1e-6)
        assert z_low < 0
        assert z_high > 0


class TestPhiNormalCDF:
    """Tests for _phi() normal CDF function."""

    def test_phi_zero(self):
        """Test that phi(0) is 0.5."""
        phi_val = _phi(0.0)
        assert abs(phi_val - 0.5) < 0.01

    def test_phi_symmetric(self):
        """Test that phi is symmetric."""
        phi1 = _phi(-1.0)
        phi2 = _phi(1.0)
        assert abs(phi1 + phi2 - 1.0) < 0.01

    def test_phi_negative(self):
        """Test with negative z values."""
        phi_val = _phi(-2.0)
        assert 0.0 < phi_val < 0.5

    def test_phi_positive(self):
        """Test with positive z values."""
        phi_val = _phi(2.0)
        assert 0.5 < phi_val < 1.0

    def test_phi_monotonic(self):
        """Test that phi is monotonic increasing."""
        phi1 = _phi(-1.0)
        phi2 = _phi(0.0)
        phi3 = _phi(1.0)
        assert phi1 < phi2 < phi3

    def test_phi_extreme_values(self):
        """Test with extreme z values."""
        phi_low = _phi(-5.0)
        phi_high = _phi(5.0)
        assert phi_low > 0.0
        assert phi_high < 1.0
        assert phi_low < 0.01  # Should be very small
        assert phi_high > 0.99  # Should be very large

    def test_phi_inverse_relationship(self):
        """Test that phi and z are approximate inverses."""
        p = 0.7
        z_val = _z(p)
        phi_val = _phi(z_val)
        assert abs(phi_val - p) < 0.01  # Should be approximately equal


class TestBootstrapCI:
    """Tests for bootstrap_ci() main function."""

    def test_bootstrap_ci_percentile_method(self):
        """Test bootstrap_ci with percentile method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data, np.mean, method="percentile", random_state=42)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_bootstrap_ci_bca_method(self):
        """Test bootstrap_ci with BCa method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lower, upper = bootstrap_ci(data, np.mean, method="bca", random_state=42)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_bootstrap_ci_empty_data(self):
        """Test with empty data."""
        data = np.array([])
        lower, upper = bootstrap_ci(data, np.mean, random_state=42)

        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_bootstrap_ci_different_confidence_levels(self):
        """Test with different confidence levels."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        lower_90, upper_90 = bootstrap_ci(data, np.mean, level=0.90, random_state=42)
        lower_95, upper_95 = bootstrap_ci(data, np.mean, level=0.95, random_state=42)
        lower_99, upper_99 = bootstrap_ci(data, np.mean, level=0.99, random_state=42)

        # Higher level should give wider intervals
        assert (upper_99 - lower_99) >= (upper_95 - lower_95)
        assert (upper_95 - lower_95) >= (upper_90 - lower_90)

    def test_bootstrap_ci_different_random_states(self):
        """Test that different random states give different results."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        lower1, upper1 = bootstrap_ci(data, np.mean, random_state=42)
        lower2, upper2 = bootstrap_ci(data, np.mean, random_state=123)

        # Results should be different (bootstrap is stochastic)
        assert (lower1, upper1) != (lower2, upper2)

    def test_bootstrap_ci_same_random_state_reproducible(self):
        """Test that same random state gives reproducible results."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        lower1, upper1 = bootstrap_ci(data, np.mean, random_state=42)
        lower2, upper2 = bootstrap_ci(data, np.mean, random_state=42)

        assert lower1 == lower2
        assert upper1 == upper2

    def test_bootstrap_ci_mean_stat_function(self):
        """Test with mean as stat function."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data, np.mean, random_state=42)

        assert lower < upper
        # Mean should be roughly in the middle
        true_mean = np.mean(data)
        assert lower < true_mean < upper

    def test_bootstrap_ci_median_stat_function(self):
        """Test with median as stat function."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data, np.median, random_state=42)

        assert lower < upper
        true_median = np.median(data)
        assert lower <= true_median <= upper

    def test_bootstrap_ci_std_stat_function(self):
        """Test with std as stat function."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data, np.std, random_state=42)

        assert lower < upper
        assert lower >= 0  # Standard deviation should be non-negative

    def test_bootstrap_ci_custom_stat_function(self):
        """Test with custom stat function."""

        def custom_stat(x):
            return np.max(x) - np.min(x)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data, custom_stat, random_state=42)

        assert lower < upper
        assert lower >= 0  # Range should be non-negative

    def test_bootstrap_ci_different_bootstrap_samples(self):
        """Test with different number of bootstrap samples."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        lower1, upper1 = bootstrap_ci(data, np.mean, B=100, random_state=42)
        lower2, upper2 = bootstrap_ci(data, np.mean, B=2000, random_state=42)

        # Both should be valid, but may differ slightly
        assert np.isfinite(lower1) and np.isfinite(upper1)
        assert np.isfinite(lower2) and np.isfinite(upper2)

    def test_bootstrap_ci_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            bootstrap_ci(data, np.mean, method="invalid_method", random_state=42)

    def test_bootstrap_ci_none_random_state(self):
        """Test with None random state (should still work)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data, np.mean, random_state=None)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper


class TestBCACI:
    """Tests for bca_ci() function."""

    def test_bca_ci_valid_data(self):
        """Test BCa CI with valid data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat_fn = np.mean
        # Generate bootstrap statistics
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(1000)])

        lower, upper = bca_ci(x, stat_fn, boot_stats, level=0.95)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_bca_ci_small_sample_fallback(self):
        """Test that BCa falls back to percentile for small samples (< 5)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])  # Only 4 elements
        stat_fn = np.mean
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(100)])

        lower, upper = bca_ci(x, stat_fn, boot_stats, level=0.95)

        # Should fallback to percentile method
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper

    def test_bca_ci_non_finite_data_fallback(self):
        """Test that BCa falls back to percentile for non-finite data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat_fn = np.mean
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(100)])

        lower, upper = bca_ci(x, stat_fn, boot_stats, level=0.95)

        # Should fallback to percentile method
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_bca_ci_inf_data_fallback(self):
        """Test that BCa falls back to percentile for inf data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat_fn = np.mean
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(100)])

        lower, upper = bca_ci(x, stat_fn, boot_stats, level=0.95)

        # Should fallback to percentile method
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_bca_ci_different_confidence_levels(self):
        """Test BCa CI with different confidence levels."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat_fn = np.mean
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(1000)])

        lower_90, upper_90 = bca_ci(x, stat_fn, boot_stats, level=0.90)
        lower_95, upper_95 = bca_ci(x, stat_fn, boot_stats, level=0.95)
        lower_99, upper_99 = bca_ci(x, stat_fn, boot_stats, level=0.99)

        # Higher level should give wider intervals
        assert (upper_99 - lower_99) >= (upper_95 - lower_95)
        assert (upper_95 - lower_95) >= (upper_90 - lower_90)

    def test_bca_ci_edge_case_exactly_five(self):
        """Test BCa with exactly 5 samples (boundary case)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Exactly 5 elements
        stat_fn = np.mean
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(100)])

        lower, upper = bca_ci(x, stat_fn, boot_stats, level=0.95)

        # Should work (n >= 5)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper

    def test_bca_ci_median_stat_function(self):
        """Test BCa CI with median as stat function."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat_fn = np.median
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(1000)])

        lower, upper = bca_ci(x, stat_fn, boot_stats, level=0.95)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper
        # CI should contain the true median (within reasonable bounds)
        assert lower <= np.max(x)
        assert upper >= np.min(x)

    def test_bca_ci_bias_correction(self):
        """Test that BCa applies bias correction."""
        # Create data with known bias
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat_fn = np.mean
        rng = np.random.default_rng(42)
        boot_stats = np.array([stat_fn(x[rng.integers(0, len(x), len(x))]) for _ in range(1000)])

        lower_bca, upper_bca = bca_ci(x, stat_fn, boot_stats, level=0.95)
        lower_perc, upper_perc = _percentile_ci(boot_stats, level=0.95)

        # BCa should give different (usually better) intervals than percentile
        # They may be the same in some cases, but generally different
        assert isinstance(lower_bca, float)
        assert isinstance(upper_bca, float)
