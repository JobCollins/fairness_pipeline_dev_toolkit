"""
Unit tests for effect size calculations.
"""

from __future__ import annotations

import numpy as np

from fairness_pipeline_dev_toolkit.stats.effect_size import cohens_d, risk_ratio


class TestRiskRatio:
    """Tests for risk_ratio function."""

    def test_basic_risk_ratio(self):
        """Test basic risk ratio calculation."""
        result = risk_ratio(0.6, 0.4)

        assert abs(result - 1.5) < 1e-10  # 0.6 / 0.4 = 1.5 (allow for floating point precision)

    def test_risk_ratio_one(self):
        """Test risk ratio of 1.0 (equal rates)."""
        result = risk_ratio(0.5, 0.5)

        assert result == 1.0

    def test_risk_ratio_less_than_one(self):
        """Test risk ratio less than 1.0."""
        result = risk_ratio(0.2, 0.4)

        assert result == 0.5

    def test_risk_ratio_greater_than_one(self):
        """Test risk ratio greater than 1.0."""
        result = risk_ratio(0.8, 0.2)

        assert result == 4.0

    def test_zero_denominator_returns_nan(self):
        """Test that zero denominator returns NaN."""
        result = risk_ratio(0.5, 0.0)

        assert np.isnan(result)

    def test_none_input_returns_nan(self):
        """Test that None input returns NaN."""
        result1 = risk_ratio(None, 0.5)
        result2 = risk_ratio(0.5, None)

        assert np.isnan(result1)
        assert np.isnan(result2)

    def test_infinite_input_returns_nan(self):
        """Test that infinite input returns NaN."""
        result1 = risk_ratio(np.inf, 0.5)
        result2 = risk_ratio(0.5, np.inf)

        assert np.isnan(result1)
        assert np.isnan(result2)

    def test_nan_input_returns_nan(self):
        """Test that NaN input returns NaN."""
        result1 = risk_ratio(np.nan, 0.5)
        result2 = risk_ratio(0.5, np.nan)

        assert np.isnan(result1)
        assert np.isnan(result2)

    def test_zero_numerator(self):
        """Test with zero numerator."""
        result = risk_ratio(0.0, 0.5)

        assert result == 0.0

    def test_very_small_values(self):
        """Test with very small values."""
        result = risk_ratio(0.0001, 0.0002)

        assert result == 0.5


class TestCohensD:
    """Tests for cohens_d function."""

    def test_basic_cohens_d(self):
        """Test basic Cohen's d calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])

        result = cohens_d(x, y)

        # y has higher mean, so d should be negative
        assert result < 0
        assert np.isfinite(result)

    def test_cohens_d_zero_effect(self):
        """Test Cohen's d with no effect (same means)."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = cohens_d(x, y)

        assert abs(result) < 0.01  # Should be approximately 0

    def test_cohens_d_positive_effect(self):
        """Test Cohen's d with positive effect (x > y)."""
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])

        result = cohens_d(x, y)

        assert result > 0  # x has higher mean

    def test_cohens_d_negative_effect(self):
        """Test Cohen's d with negative effect (x < y)."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])

        result = cohens_d(x, y)

        assert result < 0  # y has higher mean

    def test_cohens_d_insufficient_data_returns_nan(self):
        """Test that insufficient data returns NaN."""
        x = np.array([1])
        y = np.array([2, 3])

        result = cohens_d(x, y)

        assert np.isnan(result)

    def test_cohens_d_both_insufficient_returns_nan(self):
        """Test that both groups with insufficient data returns NaN."""
        x = np.array([1])
        y = np.array([2])

        result = cohens_d(x, y)

        assert np.isnan(result)

    def test_cohens_d_empty_arrays_returns_nan(self):
        """Test that empty arrays return NaN."""
        x = np.array([])
        y = np.array([1, 2, 3])

        result = cohens_d(x, y)

        assert np.isnan(result)

    def test_cohens_d_zero_variance_returns_nan(self):
        """Test that zero variance returns NaN."""
        x = np.array([5, 5, 5, 5, 5])  # All same value
        y = np.array([10, 10, 10, 10, 10])  # All same value

        result = cohens_d(x, y)

        # With zero variance, pooled_std is 0, should return NaN
        assert np.isnan(result)

    def test_cohens_d_with_nan_values(self):
        """Test that NaN values are handled."""
        x = np.array([1, 2, 3, np.nan, 5])
        y = np.array([6, 7, 8, 9, 10])

        result = cohens_d(x, y)

        # Should handle NaN gracefully (may compute on non-NaN values)
        assert isinstance(result, (float, type(np.nan)))

    def test_cohens_d_list_input(self):
        """Test that list input is converted to array."""
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]

        result = cohens_d(x, y)

        assert np.isfinite(result)

    def test_cohens_d_different_sizes(self):
        """Test with different group sizes."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12])  # Smaller group

        result = cohens_d(x, y)

        assert np.isfinite(result)

    def test_cohens_d_large_effect(self):
        """Test with large effect size."""
        x = np.array([100, 101, 102, 103, 104])
        y = np.array([1, 2, 3, 4, 5])

        result = cohens_d(x, y)

        assert abs(result) > 1.0  # Large effect

    def test_cohens_d_small_effect(self):
        """Test with small effect size."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        result = cohens_d(x, y)

        assert abs(result) < 1.0  # Small effect
