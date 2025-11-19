"""
Unit tests for multiple comparison corrections.
"""

from __future__ import annotations

import numpy as np

from fairness_pipeline_dev_toolkit.stats.multipletests import (
    benjamini_hochberg,
    bonferroni,
)


class TestBonferroni:
    """Tests for bonferroni function."""

    def test_basic_bonferroni(self):
        """Test basic Bonferroni correction."""
        pvals = np.array([0.01, 0.02, 0.03])

        result = bonferroni(pvals)

        assert len(result) == 3
        # p_adj = min(1, p * n_tests)
        assert result[0] == min(1.0, 0.01 * 3)  # 0.03
        assert result[1] == min(1.0, 0.02 * 3)  # 0.06
        assert result[2] == min(1.0, 0.03 * 3)  # 0.09

    def test_bonferroni_single_pvalue(self):
        """Test Bonferroni with single p-value."""
        pvals = np.array([0.05])

        result = bonferroni(pvals)

        assert len(result) == 1
        assert result[0] == 0.05  # n_tests = 1, so no change

    def test_bonferroni_caps_at_one(self):
        """Test that Bonferroni caps adjusted p-values at 1.0."""
        pvals = np.array([0.4, 0.5, 0.6])

        result = bonferroni(pvals)

        assert all(r <= 1.0 for r in result)
        # Some may exceed 1.0 before capping
        assert all(r >= 0.0 for r in result)

    def test_bonferroni_list_input(self):
        """Test that list input is converted to array."""
        pvals = [0.01, 0.02, 0.03]

        result = bonferroni(pvals)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_bonferroni_empty_array(self):
        """Test with empty array."""
        pvals = np.array([])

        result = bonferroni(pvals)

        assert len(result) == 0

    def test_bonferroni_very_small_pvalues(self):
        """Test with very small p-values."""
        pvals = np.array([0.001, 0.002, 0.003])

        result = bonferroni(pvals)

        assert all(r > 0.0 for r in result)
        assert all(r <= 1.0 for r in result)

    def test_bonferroni_many_tests(self):
        """Test with many tests (strong correction)."""
        pvals = np.array([0.01] * 100)

        result = bonferroni(pvals)

        # All should be adjusted to 1.0 (0.01 * 100 = 1.0)
        assert all(r == 1.0 for r in result)


class TestBenjaminiHochberg:
    """Tests for benjamini_hochberg function."""

    def test_basic_benjamini_hochberg(self):
        """Test basic Benjamini-Hochberg correction."""
        pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        adjusted, order = benjamini_hochberg(pvals)

        assert len(adjusted) == 5
        assert len(order) == 5
        # Adjusted p-values should be monotonic (non-decreasing)
        assert all(adjusted[i] <= adjusted[i + 1] for i in range(len(adjusted) - 1))
        # All should be <= 1.0
        assert all(p <= 1.0 for p in adjusted)
        assert all(p >= 0.0 for p in adjusted)

    def test_benjamini_hochberg_single_pvalue(self):
        """Test Benjamini-Hochberg with single p-value."""
        pvals = np.array([0.05])

        adjusted, order = benjamini_hochberg(pvals)

        assert len(adjusted) == 1
        assert len(order) == 1
        assert adjusted[0] == 0.05  # No correction for single test
        assert order[0] == 0

    def test_benjamini_hochberg_order_preservation(self):
        """Test that order indices are correct."""
        pvals = np.array([0.05, 0.01, 0.03, 0.02, 0.04])

        adjusted, order = benjamini_hochberg(pvals)

        # Order should indicate original positions
        assert len(order) == 5
        assert set(order) == {0, 1, 2, 3, 4}

    def test_benjamini_hochberg_caps_at_one(self):
        """Test that adjusted p-values are capped at 1.0."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        adjusted, order = benjamini_hochberg(pvals)

        assert all(p <= 1.0 for p in adjusted)
        assert all(p >= 0.0 for p in adjusted)

    def test_benjamini_hochberg_list_input(self):
        """Test that list input is converted to array."""
        pvals = [0.01, 0.02, 0.03]

        adjusted, order = benjamini_hochberg(pvals)

        assert isinstance(adjusted, np.ndarray)
        assert isinstance(order, np.ndarray)

    def test_benjamini_hochberg_empty_array(self):
        """Test with empty array."""
        pvals = np.array([])

        adjusted, order = benjamini_hochberg(pvals)

        assert len(adjusted) == 0
        assert len(order) == 0

    def test_benjamini_hochberg_monotonicity(self):
        """Test that adjusted p-values are monotonic."""
        pvals = np.array([0.01, 0.05, 0.02, 0.03, 0.04])

        adjusted, order = benjamini_hochberg(pvals)

        # After sorting, adjusted should be monotonic
        assert all(adjusted[i] <= adjusted[i + 1] for i in range(len(adjusted) - 1))

    def test_benjamini_hochberg_sorted_input(self):
        """Test with already sorted p-values."""
        pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        adjusted, order = benjamini_hochberg(pvals)

        # Order should be [0, 1, 2, 3, 4] for sorted input
        assert np.array_equal(order, np.array([0, 1, 2, 3, 4]))

    def test_benjamini_hochberg_reverse_sorted(self):
        """Test with reverse sorted p-values."""
        pvals = np.array([0.05, 0.04, 0.03, 0.02, 0.01])

        adjusted, order = benjamini_hochberg(pvals)

        # Should still work correctly
        assert len(adjusted) == 5
        assert len(order) == 5

    def test_benjamini_hochberg_very_small_pvalues(self):
        """Test with very small p-values."""
        pvals = np.array([0.001, 0.002, 0.003])

        adjusted, order = benjamini_hochberg(pvals)

        assert all(p > 0.0 for p in adjusted)
        assert all(p <= 1.0 for p in adjusted)

    def test_benjamini_hochberg_many_tests(self):
        """Test with many tests."""
        pvals = np.array([0.01] * 50)

        adjusted, order = benjamini_hochberg(pvals)

        assert len(adjusted) == 50
        assert len(order) == 50
        # All p-values are the same, so adjusted should be the same
        assert all(p == adjusted[0] for p in adjusted)
