"""
Tests for FairlearnAdapter class.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.metrics.fairlearn_adapter import FairlearnAdapter


class TestFairlearnAdapterInit:
    """Tests for FairlearnAdapter.__init__() and available()."""

    def test_init_when_fairlearn_available(self):
        """Test __init__ when fairlearn is available."""
        # Create a mock fairlearn.metrics module
        mock_fairlearn_metrics = MagicMock()
        mock_fairlearn = MagicMock()
        mock_fairlearn.metrics = mock_fairlearn_metrics

        # Patch sys.modules to include the mock
        with patch.dict(
            sys.modules, {"fairlearn": mock_fairlearn, "fairlearn.metrics": mock_fairlearn_metrics}
        ):
            adapter = FairlearnAdapter()
            assert adapter.available() is True
            assert adapter._ok is True

    def test_init_when_fairlearn_unavailable(self):
        """Test __init__ when fairlearn is not available."""

        # Mock fairlearn import to fail
        def mock_import(name, *args, **kwargs):
            if name == "fairlearn.metrics" or name == "fairlearn":
                raise ImportError("No module named 'fairlearn'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = FairlearnAdapter()
            assert adapter.available() is False
            assert adapter._ok is False


class TestMAEParityDifference:
    """Tests for mae_parity_difference() method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with mocked fairlearn."""
        # Create a mock fairlearn.metrics module
        mock_fairlearn_metrics = MagicMock()
        mock_fairlearn = MagicMock()
        mock_fairlearn.metrics = mock_fairlearn_metrics

        # Patch sys.modules to include the mock
        with patch.dict(
            sys.modules, {"fairlearn": mock_fairlearn, "fairlearn.metrics": mock_fairlearn_metrics}
        ):
            return FairlearnAdapter()

    def test_mae_valid_groups(self, adapter):
        """Test with valid groups (>= min_group_size)."""
        y_true = np.array([3.0, 2.5, 4.0, 5.0, 3.5, 4.5])
        y_pred = np.array([2.8, 2.3, 4.2, 5.1, 3.2, 4.3])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "mae_parity_difference"
        assert not np.isnan(result.value)
        assert result.value >= 0  # Difference should be non-negative
        assert result.n_per_group is not None
        assert len(result.n_per_group) == 2  # Two groups

    def test_mae_insufficient_groups(self, adapter):
        """Test with insufficient groups (< min_group_size)."""
        y_true = np.array([3.0, 2.5, 4.0, 5.0])
        y_pred = np.array([2.8, 2.3, 4.2, 5.1])
        sensitive = np.array(["A", "A", "B", "B"])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=3)

        assert isinstance(result, MetricResult)
        assert result.metric == "mae_parity_difference"
        assert np.isnan(result.value)  # Should return NaN when no valid groups
        assert result.n_per_group == {}

    def test_mae_empty_data(self, adapter):
        """Test with empty data."""
        y_true = np.array([])
        y_pred = np.array([])
        sensitive = np.array([])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.metric == "mae_parity_difference"
        assert np.isnan(result.value)
        assert result.n_per_group == {}

    def test_mae_single_group(self, adapter):
        """Test with single group."""
        y_true = np.array([3.0, 2.5, 4.0, 3.5])
        y_pred = np.array([2.8, 2.3, 4.2, 3.2])
        sensitive = np.array(["A", "A", "A", "A"])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        # Need at least 2 groups to compute difference
        assert np.isnan(result.value)

    def test_mae_perfect_parity(self, adapter):
        """Test with perfect MAE parity (same MAE for all groups)."""
        # Perfect predictions for both groups
        y_true = np.array([3.0, 2.5, 4.0, 3.0, 2.5, 4.0])
        y_pred = np.array([3.0, 2.5, 4.0, 3.0, 2.5, 4.0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value == 0.0  # Perfect parity means 0 difference

    def test_mae_different_errors(self, adapter):
        """Test with different MAE values across groups."""
        # Group A: small errors, Group B: larger errors
        y_true = np.array([3.0, 3.0, 3.0, 5.0, 5.0, 5.0])
        y_pred = np.array([3.1, 2.9, 3.0, 4.5, 5.5, 4.8])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert isinstance(result, MetricResult)
        assert result.value > 0  # Should have some difference

    def test_mae_unavailable_raises_error(self):
        """Test that calling method when fairlearn unavailable raises RuntimeError."""

        def mock_import(name, *args, **kwargs):
            if name == "fairlearn.metrics" or name == "fairlearn":
                raise ImportError("No module named 'fairlearn'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            adapter = FairlearnAdapter()
            y_true = np.array([3.0, 2.5])
            y_pred = np.array([2.8, 2.3])
            sensitive = np.array(["A", "B"])

            with pytest.raises(RuntimeError, match="Fairlearn not available"):
                adapter.mae_parity_difference(y_true, y_pred, sensitive)

    def test_mae_n_per_group_counts(self, adapter):
        """Test that n_per_group contains correct counts."""
        y_true = np.array([3.0, 2.5, 4.0, 5.0, 3.5, 4.5])
        y_pred = np.array([2.8, 2.3, 4.2, 5.1, 3.2, 4.3])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = adapter.mae_parity_difference(y_true, y_pred, sensitive, min_group_size=2)

        assert result.n_per_group["A"] == 3
        assert result.n_per_group["B"] == 3
