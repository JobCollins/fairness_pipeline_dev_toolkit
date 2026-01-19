"""
Tests for AequitasAdapter class.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.metrics.aequitas_adapter import AequitasAdapter
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


class TestAequitasAdapterInit:
    """Tests for AequitasAdapter.__init__() and available()."""

    def test_init_when_aequitas_available(self):
        """Test __init__ when aequitas is available."""

        # Mock aequitas import to succeed
        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                return type("MockModule", (), {})()
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = AequitasAdapter()
            assert adapter.available() is True
            assert adapter._ok is True

    def test_init_when_aequitas_unavailable(self):
        """Test __init__ when aequitas is not available."""

        # Mock aequitas import to fail
        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                raise ImportError("No module named 'aequitas'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = AequitasAdapter()
            assert adapter.available() is False
            assert adapter._ok is False

    def test_init_when_aequitas_raises_other_exception(self):
        """Test __init__ when aequitas import raises non-ImportError."""

        # Mock aequitas import to raise a different exception
        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                raise RuntimeError("Some other error")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = AequitasAdapter()
            assert adapter.available() is False
            assert adapter._ok is False

    def test_available_returns_bool(self):
        """Test that available() always returns a boolean."""

        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                return type("MockModule", (), {})()
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = AequitasAdapter()
            result = adapter.available()
            assert isinstance(result, bool)


class TestMaskSmallGroups:
    """Tests for _mask_small_groups() method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with mocked aequitas."""

        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                return type("MockModule", (), {})()
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            return AequitasAdapter()

    def test_mask_small_groups_all_valid(self, adapter):
        """Test with all groups meeting min_group_size."""
        sensitive = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        assert valid.all()  # All groups should be valid

    def test_mask_small_groups_some_invalid(self, adapter):
        """Test with some groups below min_group_size."""
        sensitive = ["A", "A", "A", "B", "C"]  # B and C have only 1
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        # A should be valid (3 >= 2), B and C should be invalid (1 < 2)
        assert valid[0]  # A
        assert valid[1]  # A
        assert valid[2]  # A
        assert not valid[3]  # B
        assert not valid[4]  # C

    def test_mask_small_groups_all_invalid(self, adapter):
        """Test with all groups below min_group_size."""
        sensitive = ["A", "B", "C"]  # Each has only 1
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        assert not valid.any()  # No groups should be valid

    def test_mask_small_groups_exact_threshold(self, adapter):
        """Test with groups exactly at min_group_size."""
        sensitive = ["A", "A", "B", "B"]  # Each has exactly 2
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        assert valid.all()  # All should be valid (>= threshold)

    def test_mask_small_groups_empty(self, adapter):
        """Test with empty sensitive array."""
        sensitive = []
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(s) == 0
        assert len(valid) == 0

    def test_mask_small_groups_single_group(self, adapter):
        """Test with single group."""
        sensitive = ["A", "A", "A", "A", "A"]
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=3)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        assert valid.all()  # All should be valid

    def test_mask_small_groups_pandas_series(self, adapter):
        """Test with pandas Series input."""
        sensitive = pd.Series(["A", "A", "B", "B", "C"])
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        # A and B should be valid, C should be invalid
        assert valid[0]  # A
        assert valid[1]  # A
        assert valid[2]  # B
        assert valid[3]  # B
        assert not valid[4]  # C

    def test_mask_small_groups_numpy_array(self, adapter):
        """Test with numpy array input."""
        sensitive = np.array(["A", "A", "B", "B", "C"])
        s, valid = adapter._mask_small_groups(sensitive, min_group_size=2)

        assert isinstance(s, pd.Series)
        assert len(valid) == len(sensitive)
        assert isinstance(valid, np.ndarray)


class TestDemographicParityDifference:
    """Tests for demographic_parity_difference() method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with mocked aequitas."""

        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                return type("MockModule", (), {})()
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            return AequitasAdapter()

    def test_dp_valid_groups(self, adapter):
        """Test with valid groups (>= min_group_size)."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "demographic_parity_difference"
        assert not np.isnan(result.value)
        assert result.value >= 0  # Difference should be non-negative
        assert result.n_per_group is not None
        assert len(result.n_per_group) == 2  # Two groups

    def test_dp_insufficient_groups(self, adapter):
        """Test with insufficient groups (< min_group_size)."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        sensitive = np.array(["A", "A", "B", "B"])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=3)

        assert isinstance(result, MetricResult)
        assert result.metric == "demographic_parity_difference"
        assert np.isnan(result.value)  # Should return NaN when no valid groups
        assert result.n_per_group == {}

    def test_dp_empty_data(self, adapter):
        """Test with empty data."""
        y_true = np.array([])
        y_pred = np.array([])
        sensitive = np.array([])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "demographic_parity_difference"
        assert np.isnan(result.value)
        assert result.n_per_group == {}

    def test_dp_single_group(self, adapter):
        """Test with single group."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "A"])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "demographic_parity_difference"
        assert np.isnan(result.value)  # Need at least 2 groups
        assert result.n_per_group is not None

    def test_dp_less_than_two_groups_after_filtering(self, adapter):
        """Test when filtering leaves less than 2 groups."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B"])  # B has only 2

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=3)

        assert isinstance(result, MetricResult)
        assert np.isnan(result.value)  # Only A group remains after filtering
        assert result.n_per_group is not None

    def test_dp_perfect_parity(self, adapter):
        """Test with perfect demographic parity (same rates)."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # All predictions are 1
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value == 0.0  # Perfect parity means 0 difference

    def test_dp_maximum_difference(self, adapter):
        """Test with maximum difference (one group all 0, other all 1)."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # A: all 0, B: all 1
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value == 1.0  # Maximum difference

    def test_dp_unavailable_raises_error(self):
        """Test that calling method when aequitas unavailable raises RuntimeError."""

        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                raise ImportError("No module named 'aequitas'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = AequitasAdapter()
            y_true = np.array([0, 1])
            y_pred = np.array([0, 1])
            sensitive = np.array(["A", "B"])

            with pytest.raises(RuntimeError, match="Aequitas not available"):
                adapter.demographic_parity_difference(y_true, y_pred, sensitive)

    def test_dp_n_per_group_counts(self, adapter):
        """Test that n_per_group contains correct counts."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert result.n_per_group["A"] == 4
        assert result.n_per_group["B"] == 4


class TestEqualizedOddsDifference:
    """Tests for equalized_odds_difference() method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with mocked aequitas."""

        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                return type("MockModule", (), {})()
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            return AequitasAdapter()

    def test_eo_valid_groups(self, adapter):
        """Test with valid groups and both TPR and FPR available."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "equalized_odds_difference"
        assert not np.isnan(result.value)
        assert result.value >= 0  # Difference should be non-negative
        assert result.n_per_group is not None
        assert len(result.n_per_group) == 2

    def test_eo_insufficient_groups(self, adapter):
        """Test with insufficient groups (< min_group_size)."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        sensitive = np.array(["A", "A", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=3)

        assert isinstance(result, MetricResult)
        assert result.metric == "equalized_odds_difference"
        assert np.isnan(result.value)
        assert result.n_per_group == {}

    def test_eo_empty_data(self, adapter):
        """Test with empty data."""
        y_true = np.array([])
        y_pred = np.array([])
        sensitive = np.array([])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "equalized_odds_difference"
        assert np.isnan(result.value)
        assert result.n_per_group == {}

    def test_eo_missing_tpr_no_positives(self, adapter):
        """Test when a group has no positive labels (TPR is NaN)."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # A: all 0, B: all 1
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        # Group A has no positives, so TPR is NaN, which makes result NaN
        assert np.isnan(result.value)

    def test_eo_missing_fpr_no_negatives(self, adapter):
        """Test when a group has no negative labels (FPR is NaN)."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # A: all 1, B: all 0
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        # Group A has no negatives, so FPR is NaN, which makes result NaN
        assert np.isnan(result.value)

    def test_eo_single_group(self, adapter):
        """Test with single group."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "A"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        # Need at least 2 groups to compute difference
        assert np.isnan(result.value)

    def test_eo_perfect_equality(self, adapter):
        """Test with perfect equalized odds (same TPR and FPR)."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])  # Perfect predictions
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value == 0.0  # Perfect equality means 0 difference

    def test_eo_tpr_difference_dominates(self, adapter):
        """Test when TPR difference is larger than FPR difference."""
        # Group A: TPR=1.0 (2/2), FPR=0.0 (0/2); Group B: TPR=0.5 (1/2), FPR=0.0 (0/2)
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])  # A: [1,1,0,0], B: [1,1,0,0]
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])  # A: perfect, B: misses one positive
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value > 0  # Should have some difference (TPR gap = 0.5)

    def test_eo_fpr_difference_dominates(self, adapter):
        """Test when FPR difference is larger than TPR difference."""
        # Group A: TPR=1.0 (2/2), FPR=0.5 (1/2); Group B: TPR=1.0 (2/2), FPR=0.0 (0/2)
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0])  # A: false positive, B: none
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value > 0  # Should have some difference (FPR gap = 0.5)

    def test_eo_unavailable_raises_error(self):
        """Test that calling method when aequitas unavailable raises RuntimeError."""

        def mock_import(name, *args, **kwargs):
            if name == "aequitas":
                raise ImportError("No module named 'aequitas'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = AequitasAdapter()
            y_true = np.array([0, 1])
            y_pred = np.array([0, 1])
            sensitive = np.array(["A", "B"])

            with pytest.raises(RuntimeError, match="Aequitas not available"):
                adapter.equalized_odds_difference(y_true, y_pred, sensitive)

    def test_eo_n_per_group_counts(self, adapter):
        """Test that n_per_group contains correct counts."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert result.n_per_group["A"] == 4
        assert result.n_per_group["B"] == 4

    def test_eo_less_than_two_valid_tpr(self, adapter):
        """Test when less than 2 groups have valid TPR values."""
        # Group A: no positives (TPR=NaN), Group B: has positives
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        # Only one group has valid TPR, so span returns NaN
        assert np.isnan(result.value)

    def test_eo_less_than_two_valid_fpr(self, adapter):
        """Test when less than 2 groups have valid FPR values."""
        # Group A: no negatives (FPR=NaN), Group B: has negatives
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        # Only one group has valid FPR, so span returns NaN
        assert np.isnan(result.value)
