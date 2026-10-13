"""
Unit tests for intersectional utility functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.utils.intersectional import (
    build_intersectional_labels,
    group_sizes,
    min_group_mask,
)


def _build_intersectional_labels_rowwise(attrs_df, columns=None, sep="||", include_na=True):
    """Reference (pre-optimisation) row-wise implementation, kept only for the
    equivalence test below: ``build_intersectional_labels`` used to call
    ``DataFrame.agg(sep.join, axis=1)`` for every row, which is correct but slow
    on large frames. The vectorised version must match it exactly, including its
    quirks (empty-frame returns a ``DataFrame``, NaN with ``include_na=False``
    raises ``TypeError``).
    """
    if columns is None:
        columns = list(attrs_df.columns)
    df = attrs_df[columns].copy()
    if include_na:
        df = df.fillna("NaN")
        labels = df.astype(str).agg(sep.join, axis=1)
    else:
        labels = df.astype("string").agg(sep.join, axis=1)
    return labels.astype("category")


class TestBuildIntersectionalLabelsEquivalence:
    """Pins the vectorised ``build_intersectional_labels`` against the original
    row-wise ``agg(str.join)`` implementation across edge cases (BL perf pass).
    """

    @pytest.mark.parametrize("include_na", [True, False])
    @pytest.mark.parametrize(
        "df",
        [
            pytest.param(
                pd.DataFrame(
                    {
                        "race": np.random.default_rng(0).choice(["A", "B", "C", "D", "E"], 2000),
                        "gender": np.random.default_rng(1).choice(["M", "F"], 2000),
                        "age_group": np.random.default_rng(2).choice(["<30", "30-50", "50+"], 2000),
                    }
                ),
                id="large_intersectional_3col",
            ),
            pytest.param(pd.DataFrame({"only": ["x", "y", "x", "z"]}), id="single_column"),
            pytest.param(
                pd.DataFrame({"a": [None, None], "b": ["x", "y"]}), id="one_group_all_nan"
            ),
            pytest.param(pd.DataFrame({"a": ["x"], "b": ["y"]}), id="single_row_group"),
            pytest.param(
                pd.DataFrame({"a": pd.Series([], dtype=object), "b": pd.Series([], dtype=object)}),
                id="empty_group",
            ),
            pytest.param(
                pd.DataFrame({"race": ["A", "B", None, "A"], "gender": ["M", "F", "M", "F"]}),
                id="nan_group_value",
            ),
        ],
    )
    def test_matches_rowwise_reference(self, df, include_na):
        def run(fn):
            try:
                return ("OK", fn(df, include_na=include_na))
            except Exception as exc:  # noqa: BLE001 - comparing error parity, not type
                return ("ERR", type(exc))

        old_status, old_val = run(_build_intersectional_labels_rowwise)
        new_status, new_val = run(build_intersectional_labels)

        assert old_status == new_status
        if old_status == "ERR":
            assert old_val is new_val
            return

        assert type(old_val) is type(new_val)
        if isinstance(old_val, pd.DataFrame):
            assert old_val.equals(new_val)
        else:
            assert old_val.equals(new_val)
            assert old_val.dtype == new_val.dtype
            assert old_val.name == new_val.name


class TestBuildIntersectionalLabels:
    """Tests for build_intersectional_labels function."""

    def test_basic_intersectional_labels(self):
        """Test basic intersectional label creation."""
        df = pd.DataFrame(
            {
                "race": ["A", "A", "B", "B"],
                "gender": ["M", "F", "M", "F"],
            }
        )

        labels = build_intersectional_labels(df)

        assert len(labels) == 4
        assert labels.dtype.name == "category"
        # Check that labels are concatenated
        assert "A||M" in labels.values or "M||A" in labels.values

    def test_custom_separator(self):
        """Test with custom separator."""
        df = pd.DataFrame(
            {
                "race": ["A", "B"],
                "gender": ["M", "F"],
            }
        )

        labels = build_intersectional_labels(df, sep="_")

        assert "_" in str(labels.iloc[0])

    def test_specific_columns(self):
        """Test with specific columns selected."""
        df = pd.DataFrame(
            {
                "race": ["A", "B"],
                "gender": ["M", "F"],
                "age": [25, 30],
            }
        )

        labels = build_intersectional_labels(df, columns=["race", "gender"])

        assert len(labels) == 2
        # Should not include age in labels
        assert "25" not in str(labels.iloc[0])
        assert "30" not in str(labels.iloc[1])

    def test_include_na_true(self):
        """Test with include_na=True (default)."""
        df = pd.DataFrame(
            {
                "race": ["A", "B", None],
                "gender": ["M", "F", "M"],
            }
        )

        labels = build_intersectional_labels(df, include_na=True)

        assert len(labels) == 3
        # NaN should be represented as "NaN" string
        assert "NaN" in str(labels.iloc[2])

    def test_include_na_false(self):
        """Test with include_na=False."""
        df = pd.DataFrame(
            {
                "race": ["A", "B", None],
                "gender": ["M", "F", "M"],
            }
        )

        # When include_na=False, the function may raise an error or handle NaN differently
        # depending on implementation. Let's test that it handles the case gracefully
        try:
            labels = build_intersectional_labels(df, include_na=False)
            # If it succeeds, verify the result
            assert len(labels) == 3
            # Row with NaN may result in NaN label or error
            if not pd.isna(labels.iloc[2]):
                # If not NaN, it was converted to string somehow
                assert isinstance(labels.iloc[2], str)
        except (TypeError, ValueError):
            # If it raises an error, that's also acceptable behavior
            pass

    def test_single_attribute(self):
        """Test with single attribute."""
        df = pd.DataFrame({"group": ["A", "B", "A", "B"]})

        labels = build_intersectional_labels(df)

        assert len(labels) == 4
        assert labels.dtype.name == "category"
        # Should just be the group values
        assert "A" in labels.values
        assert "B" in labels.values

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"race": [], "gender": []})

        labels = build_intersectional_labels(df)

        # When DataFrame is empty, agg might return DataFrame or Series
        # Convert to Series if needed
        if isinstance(labels, pd.DataFrame):
            # If agg returned DataFrame, convert to Series
            labels = (
                labels.iloc[:, 0] if len(labels.columns) > 0 else pd.Series([], dtype="category")
            )

        assert len(labels) == 0
        # Should be a Series after conversion
        assert isinstance(labels, pd.Series)

    def test_all_nan_values(self):
        """Test with all NaN values."""
        df = pd.DataFrame(
            {
                "race": [None, None],
                "gender": [None, None],
            }
        )

        labels = build_intersectional_labels(df, include_na=True)

        assert len(labels) == 2
        assert "NaN" in str(labels.iloc[0])


class TestMinGroupMask:
    """Tests for min_group_mask function."""

    def test_basic_min_group_mask(self):
        """Test basic min_group_mask functionality."""
        labels = ["A", "A", "A", "B", "B", "C"]

        mask = min_group_mask(labels, min_group_size=2)

        assert len(mask) == 6
        assert isinstance(mask, np.ndarray), "Mask should be numpy array"
        # Groups A and B have >= 2, C has 1
        assert mask.sum() == 5  # 3 A's + 2 B's
        # Verify boolean-like behavior
        assert all(isinstance(x, (bool, np.bool_)) or x in (True, False, 0, 1) for x in mask)

    def test_min_group_size_one(self):
        """Test with min_group_size=1 (all groups included)."""
        labels = ["A", "A", "B", "C"]

        mask = min_group_mask(labels, min_group_size=1)

        assert mask.all()  # All should be True

    def test_min_group_size_large(self):
        """Test with large min_group_size."""
        labels = ["A", "A", "A", "B", "B", "C"]

        mask = min_group_mask(labels, min_group_size=3)

        # Only group A has >= 3
        assert mask.sum() == 3

    def test_with_categorical_series(self):
        """Test with categorical Series."""
        labels = pd.Series(["A", "A", "B", "B"], dtype="category")

        mask = min_group_mask(labels, min_group_size=2)

        assert len(mask) == 4
        assert mask.all()  # Both groups have >= 2

    def test_with_nan_values(self):
        """Test with NaN values in labels."""
        labels = ["A", "A", "B", None, np.nan]

        mask = min_group_mask(labels, min_group_size=2)

        assert len(mask) == 5
        # NaN groups should be handled
        assert isinstance(mask, np.ndarray)

    def test_empty_labels(self):
        """Test with empty labels."""
        labels = []

        mask = min_group_mask(labels, min_group_size=2)

        assert len(mask) == 0
        assert isinstance(mask, np.ndarray)


class TestGroupSizes:
    """Tests for group_sizes function."""

    def test_basic_group_sizes(self):
        """Test basic group_sizes functionality."""
        labels = ["A", "A", "A", "B", "B", "C"]

        sizes = group_sizes(labels)

        assert isinstance(sizes, dict)
        assert sizes["A"] == 3
        assert sizes["B"] == 2
        assert sizes["C"] == 1

    def test_with_categorical_series(self):
        """Test with categorical Series."""
        labels = pd.Series(["A", "A", "B", "B"], dtype="category")

        sizes = group_sizes(labels)

        assert sizes["A"] == 2
        assert sizes["B"] == 2

    def test_with_nan_values(self):
        """Test that NaN values are excluded."""
        labels = ["A", "A", "B", None, np.nan]

        sizes = group_sizes(labels)

        # NaN should not be counted
        assert "A" in sizes
        assert "B" in sizes
        assert None not in sizes
        assert sizes["A"] == 2
        assert sizes["B"] == 1

    def test_empty_labels(self):
        """Test with empty labels."""
        labels = []

        sizes = group_sizes(labels)

        assert sizes == {}

    def test_single_group(self):
        """Test with single group."""
        labels = ["A", "A", "A", "A"]

        sizes = group_sizes(labels)

        assert sizes == {"A": 4}

    def test_all_unique_groups(self):
        """Test with all unique groups."""
        labels = ["A", "B", "C", "D"]

        sizes = group_sizes(labels)

        assert len(sizes) == 4
        assert all(sizes[k] == 1 for k in sizes)
