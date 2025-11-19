"""
Unit tests for validation utility functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.utils.validation import (
    InputSpec,
    check_lengths,
    coerce_arrays,
)


class TestCoerceArrays:
    """Tests for coerce_arrays function."""

    def test_coerce_lists_to_arrays(self):
        """Test that lists are coerced to numpy arrays."""
        y_true, y_pred, scores = coerce_arrays(
            y_true=[1, 0, 1], y_pred=[1, 1, 0], scores=[0.8, 0.6, 0.4]
        )

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(scores, np.ndarray)

    def test_coerce_numpy_arrays(self):
        """Test that numpy arrays are returned as-is."""
        y_true_arr = np.array([1, 0, 1])
        y_pred_arr = np.array([1, 1, 0])

        y_true, y_pred, scores = coerce_arrays(y_true=y_true_arr, y_pred=y_pred_arr)

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert y_true is y_true_arr  # Should be same object
        assert scores is None

    def test_none_values(self):
        """Test that None values are preserved."""
        y_true, y_pred, scores = coerce_arrays(y_true=None, y_pred=None, scores=None)

        assert y_true is None
        assert y_pred is None
        assert scores is None

    def test_partial_none_values(self):
        """Test with some None values."""
        y_true, y_pred, scores = coerce_arrays(y_true=[1, 0, 1], y_pred=None, scores=[0.8, 0.6])

        assert isinstance(y_true, np.ndarray)
        assert y_pred is None
        assert isinstance(scores, np.ndarray)

    def test_coerce_pandas_series(self):
        """Test that pandas Series are coerced to arrays."""
        y_true_series = pd.Series([1, 0, 1])
        y_pred_series = pd.Series([1, 1, 0])

        y_true, y_pred, scores = coerce_arrays(y_true=y_true_series, y_pred=y_pred_series)

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)


class TestCheckLengths:
    """Tests for check_lengths function."""

    def test_matching_lengths(self):
        """Test that matching lengths pass."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        arr3 = np.array([7, 8, 9])

        # Should not raise
        check_lengths(arr1, arr2, arr3)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])  # Different length

        with pytest.raises(ValueError, match="Mismatched lengths"):
            check_lengths(arr1, arr2)

    def test_with_none_values(self):
        """Test that None values are ignored."""
        arr1 = np.array([1, 2, 3])
        arr2 = None
        arr3 = np.array([7, 8, 9])

        # Should not raise (None is ignored)
        check_lengths(arr1, arr2, arr3)

    def test_single_array(self):
        """Test with single array."""
        arr1 = np.array([1, 2, 3])

        # Should not raise
        check_lengths(arr1)

    def test_all_none(self):
        """Test with all None values."""
        # Should not raise (no arrays to check)
        check_lengths(None, None, None)

    def test_empty_arrays(self):
        """Test with empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([])

        # Should not raise (both empty, same length)
        check_lengths(arr1, arr2)

    def test_mixed_types(self):
        """Test with mixed types (list, array, Series)."""
        arr1 = [1, 2, 3]
        arr2 = np.array([4, 5, 6])
        arr3 = pd.Series([7, 8, 9])

        # Should not raise (all same length)
        check_lengths(arr1, arr2, arr3)


class TestInputSpec:
    """Tests for InputSpec dataclass."""

    def test_input_spec_creation(self):
        """Test creating InputSpec with all fields."""
        spec = InputSpec(
            y_true=[1, 0, 1],
            y_pred=[1, 1, 0],
            scores=[0.8, 0.6, 0.4],
            attrs_df=pd.DataFrame({"group": ["A", "B", "A"]}),
        )

        assert spec.y_true == [1, 0, 1]
        assert spec.y_pred == [1, 1, 0]
        assert spec.scores == [0.8, 0.6, 0.4]
        assert isinstance(spec.attrs_df, pd.DataFrame)

    def test_input_spec_defaults(self):
        """Test InputSpec with default (None) values."""
        spec = InputSpec()

        assert spec.y_true is None
        assert spec.y_pred is None
        assert spec.scores is None
        assert spec.attrs_df is None

    def test_input_spec_partial(self):
        """Test InputSpec with partial fields."""
        spec = InputSpec(y_true=[1, 0, 1], y_pred=[1, 1, 0])

        assert spec.y_true == [1, 0, 1]
        assert spec.y_pred == [1, 1, 0]
        assert spec.scores is None
        assert spec.attrs_df is None
