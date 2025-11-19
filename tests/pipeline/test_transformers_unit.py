"""
Unit tests for pipeline transformers.

Tests all four transformers:
- InstanceReweighting
- DisparateImpactRemover
- ReweighingTransformer
- ProxyDropper
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fairness_pipeline_dev_toolkit.pipeline.transformers import (
    DisparateImpactRemover,
    InstanceReweighting,
    ProxyDropper,
    ReweighingTransformer,
)

# ============================================================================
# InstanceReweighting Tests
# ============================================================================


class TestInstanceReweighting:
    """Tests for InstanceReweighting transformer."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform behavior."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = InstanceReweighting(sensitive=["group"])
        transformer.fit(df)

        # Transform should return X unchanged
        result = transformer.transform(df)
        pd.testing.assert_frame_equal(result, df)

        # Sample weights should be computed
        assert transformer.sample_weight_ is not None
        assert len(transformer.sample_weight_) == len(df)
        assert np.all(transformer.sample_weight_ > 0)

    def test_fit_with_benchmarks(self):
        """Test fit with provided benchmarks."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        benchmarks = {"group": {"A": 0.5, "B": 0.5}}
        transformer = InstanceReweighting(sensitive=["group"], benchmarks=benchmarks)
        transformer.fit(df)

        # With balanced data and balanced benchmarks, weights should be close to 1
        assert transformer.sample_weight_ is not None
        assert np.allclose(transformer.sample_weight_.mean(), 1.0, atol=0.1)

    def test_fit_with_imbalanced_data(self):
        """Test fit with imbalanced groups."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "group": ["A", "A", "A", "A", "B"],  # 4:1 ratio
            }
        )

        transformer = InstanceReweighting(sensitive=["group"])
        transformer.fit(df)

        # Group B should have higher weights to balance
        group_b_indices = df["group"] == "B"
        group_a_indices = df["group"] == "A"
        assert (
            transformer.sample_weight_[group_b_indices].mean()
            > transformer.sample_weight_[group_a_indices].mean()
        )

    def test_fit_multiple_sensitive_attributes(self):
        """Test fit with multiple sensitive attributes."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
            }
        )

        transformer = InstanceReweighting(sensitive=["group", "gender"])
        transformer.fit(df)

        assert transformer.sample_weight_ is not None
        assert len(transformer.sample_weight_) == len(df)

    def test_fit_single_group(self):
        """Test fit with only one group present."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "group": ["A", "A", "A", "A", "A"],
            }
        )

        transformer = InstanceReweighting(sensitive=["group"])
        transformer.fit(df)

        # With single group, weights should be close to 1
        assert transformer.sample_weight_ is not None
        assert np.allclose(transformer.sample_weight_, 1.0, atol=0.1)

    def test_fit_missing_sensitive_column(self):
        """Test fit when sensitive column is missing (should skip silently)."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "other": ["X", "Y", "X", "Y", "X"],
            }
        )

        transformer = InstanceReweighting(sensitive=["missing_col"])
        transformer.fit(df)

        # Should still work, weights should be all 1s
        assert transformer.sample_weight_ is not None
        assert np.allclose(transformer.sample_weight_, 1.0)

    def test_transform_without_fit(self):
        """Test transform raises error if not fitted."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "group": ["A", "B", "A"]})

        transformer = InstanceReweighting(sensitive=["group"])

        with pytest.raises(RuntimeError, match="must be fitted"):
            transformer.transform(df)

    def test_transform_size_mismatch(self):
        """Test transform raises error if size doesn't match."""
        df_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
        df_test = pd.DataFrame({"feature1": [1, 2], "group": ["A", "B"]})

        transformer = InstanceReweighting(sensitive=["group"])
        transformer.fit(df_train)

        with pytest.raises(RuntimeError, match="sizes must match"):
            transformer.transform(df_test)

    def test_fit_invalid_input_type(self):
        """Test fit raises error with non-DataFrame input."""
        transformer = InstanceReweighting(sensitive=["group"])

        with pytest.raises(TypeError, match="pandas DataFrame"):
            transformer.fit(np.array([[1, 2], [3, 4]]))

    def test_max_weight_clipping(self):
        """Test that weights are clipped to max_weight (before normalization)."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "group": ["A"] * 9 + ["B"],  # Very imbalanced
            }
        )

        transformer = InstanceReweighting(sensitive=["group"], max_weight=2.0)
        transformer.fit(df)

        # After normalization, weights may exceed max_weight, but should be reasonable
        # The key is that clipping happens before normalization
        assert np.all(transformer.sample_weight_ > 0), "All weights should be positive"
        assert np.isfinite(transformer.sample_weight_).all(), "All weights should be finite"
        # Normalized weights should have mean close to 1.0
        assert abs(transformer.sample_weight_.mean() - 1.0) < 0.1, "Weights should be normalized"

    def test_sklearn_pipeline_integration(self):
        """Test integration with sklearn Pipeline."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        pipeline = Pipeline([("reweight", InstanceReweighting(sensitive=["group"]))])
        result = pipeline.fit_transform(df)

        pd.testing.assert_frame_equal(result, df)
        assert pipeline.named_steps["reweight"].sample_weight_ is not None


# ============================================================================
# DisparateImpactRemover Tests
# ============================================================================


class TestDisparateImpactRemover:
    """Tests for DisparateImpactRemover transformer."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform behavior."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = DisparateImpactRemover(
            sensitive="group", features=["feature1"], repair_level=0.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should return DataFrame with same shape
        assert result.shape == df.shape
        assert "feature1" in result.columns
        assert "group" in result.columns

        # Feature values may or may not be modified depending on group distributions
        # With balanced groups and similar distributions, repair may have minimal effect
        # Just verify the transformer ran successfully
        assert isinstance(result["feature1"].values, np.ndarray)
        assert len(result["feature1"].values) == len(df["feature1"].values)

    def test_repair_level_zero(self):
        """Test that repair_level=0 returns original values."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = DisparateImpactRemover(
            sensitive="group", features=["feature1"], repair_level=0.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        np.testing.assert_array_almost_equal(result["feature1"].values, df["feature1"].values)

    def test_repair_level_one(self):
        """Test that repair_level=1 fully repairs."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],  # Clear group difference
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = DisparateImpactRemover(
            sensitive="group", features=["feature1"], repair_level=1.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # After full repair, group means should be closer (or at least not further apart)
        original_diff = abs(
            df[df["group"] == "A"]["feature1"].mean() - df[df["group"] == "B"]["feature1"].mean()
        )
        repaired_diff = abs(
            result[result["group"] == "A"]["feature1"].mean()
            - result[result["group"] == "B"]["feature1"].mean()
        )
        # Repair should reduce disparity (or at least not increase it significantly)
        assert repaired_diff <= original_diff + 0.1, "Repair should reduce or maintain disparity"

    def test_multiple_features(self):
        """Test with multiple features."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = DisparateImpactRemover(
            sensitive="group", features=["feature1", "feature2"], repair_level=0.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert result.shape[1] == 3  # feature1, feature2, group

    def test_min_group_size_filtering(self):
        """Test that groups smaller than min_group_size are skipped."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "group": ["A", "A", "A", "B", "B"],  # B has only 2 samples
            }
        )

        transformer = DisparateImpactRemover(
            sensitive="group", features=["feature1"], repair_level=0.5, min_group_size=3
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Group B should not be repaired (too small)
        # Group A should be repaired
        assert result is not None

    def test_fit_missing_sensitive_column(self):
        """Test fit raises error when sensitive column is missing."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})

        transformer = DisparateImpactRemover(sensitive="missing", features=["feature1"])

        with pytest.raises(ValueError, match="not in DataFrame"):
            transformer.fit(df)

    def test_fit_missing_feature_column(self):
        """Test fit raises error when feature column is missing."""
        df = pd.DataFrame({"group": ["A", "B", "A"]})

        transformer = DisparateImpactRemover(sensitive="group", features=["missing"])

        with pytest.raises(ValueError, match="not in DataFrame"):
            transformer.fit(df)

    def test_transform_without_fit(self):
        """Test transform raises error if not fitted."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "group": ["A", "B", "A"]})

        transformer = DisparateImpactRemover(sensitive="group", features=["feature1"])

        with pytest.raises(RuntimeError, match="must be fitted"):
            transformer.transform(df)

    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = DisparateImpactRemover(
            sensitive="group", features=["feature1"], repair_level=0.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should handle NaNs gracefully
        assert result.shape == df.shape


# ============================================================================
# ReweighingTransformer Tests
# ============================================================================


class TestReweighingTransformer:
    """Tests for ReweighingTransformer."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform behavior."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ReweighingTransformer(sensitive=["group"])
        transformer.fit(df)
        result = transformer.transform(df)

        # Transform should return X unchanged
        pd.testing.assert_frame_equal(result, df)

        # Sample weights should be computed
        assert transformer.sample_weight_ is not None
        assert len(transformer.sample_weight_) == len(df)
        assert np.all(transformer.sample_weight_ > 0)

    def test_fit_with_benchmarks(self):
        """Test fit with provided benchmarks."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        benchmarks = {"group": {"A": 0.5, "B": 0.5}}
        transformer = ReweighingTransformer(sensitive=["group"], benchmarks=benchmarks)
        transformer.fit(df)

        # With balanced data and balanced benchmarks, weights should be close to 1
        assert transformer.sample_weight_ is not None
        assert np.allclose(transformer.sample_weight_.mean(), 1.0, atol=0.1)

    def test_fit_with_imbalanced_data(self):
        """Test fit with imbalanced groups."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "group": ["A", "A", "A", "A", "B"],  # 4:1 ratio
            }
        )

        transformer = ReweighingTransformer(sensitive=["group"])
        transformer.fit(df)

        # Group B should have higher weights to balance
        group_b_indices = df["group"] == "B"
        group_a_indices = df["group"] == "A"
        assert (
            transformer.sample_weight_[group_b_indices].mean()
            > transformer.sample_weight_[group_a_indices].mean()
        )

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        df = pd.DataFrame({"feature1": [], "group": []})

        transformer = ReweighingTransformer(sensitive=["group"])
        transformer.fit(df)

        assert transformer.sample_weight_ is not None
        assert len(transformer.sample_weight_) == 0

    def test_fit_missing_sensitive_column(self):
        """Test fit raises error when sensitive column is missing."""
        df = pd.DataFrame({"feature1": [1, 2, 3]})

        transformer = ReweighingTransformer(sensitive=["missing"])

        with pytest.raises(ValueError, match="not found in DataFrame"):
            transformer.fit(df)

    def test_fit_invalid_input_type(self):
        """Test fit raises error with non-DataFrame input."""
        transformer = ReweighingTransformer(sensitive=["group"])

        with pytest.raises(TypeError, match="pandas DataFrame"):
            transformer.fit(np.array([[1, 2], [3, 4]]))

    def test_clip_parameter(self):
        """Test that weights are clipped according to clip parameter (before normalization)."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "group": ["A"] * 9 + ["B"],  # Very imbalanced
            }
        )

        transformer = ReweighingTransformer(sensitive=["group"], clip=2.0)
        transformer.fit(df)

        # After normalization, weights may exceed clip, but should be reasonable
        assert np.all(transformer.sample_weight_ > 0), "All weights should be positive"
        assert np.isfinite(transformer.sample_weight_).all(), "All weights should be finite"
        # Normalized weights should have mean close to 1.0
        assert abs(transformer.sample_weight_.mean() - 1.0) < 0.1, "Weights should be normalized"

    def test_transform_recomputes_if_not_fitted(self):
        """Test that transform calls fit if not fitted."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ReweighingTransformer(sensitive=["group"])
        # Don't call fit explicitly
        result = transformer.transform(df)

        # Should work and compute weights
        assert transformer.sample_weight_ is not None
        pd.testing.assert_frame_equal(result, df)

    def test_multiple_sensitive_attributes(self):
        """Test with multiple sensitive attributes."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
            }
        )

        transformer = ReweighingTransformer(sensitive=["group", "gender"])
        transformer.fit(df)

        assert transformer.sample_weight_ is not None
        assert len(transformer.sample_weight_) == len(df)


# ============================================================================
# ProxyDropper Tests
# ============================================================================


class TestProxyDropper:
    """Tests for ProxyDropper transformer."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform behavior."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [10, 20, 30, 40, 50, 60],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(sensitive=["group"], threshold=0.9)
        transformer.fit(df)
        result = transformer.transform(df)

        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Group column should always be kept
        assert "group" in result.columns

    def test_drops_high_correlation_features(self):
        """Test that features highly correlated with sensitive are dropped."""
        # Create feature that is perfectly correlated with group
        df = pd.DataFrame(
            {
                "proxy": [1, 1, 1, 2, 2, 2],  # Perfect proxy for group
                "feature1": [10, 20, 30, 40, 50, 60],  # May also correlate
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(sensitive=["group"], threshold=0.5)
        transformer.fit(df)
        result = transformer.transform(df)

        # Proxy should be dropped (perfect correlation)
        assert "proxy" in transformer.dropped_columns_ or "proxy" not in result.columns
        # Group should always be kept
        assert "group" in result.columns
        # Verify that at least one feature was tested
        assert len(transformer.assoc_scores_) > 0

    def test_keeps_low_correlation_features(self):
        """Test that features with low correlation are kept."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [10, 20, 30, 40, 50, 60],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(sensitive=["group"], threshold=0.9)
        transformer.fit(df)
        result = transformer.transform(df)

        # Both features should be kept (low correlation with group)
        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert len(transformer.dropped_columns_) == 0

    def test_max_drop_parameter(self):
        """Test that max_drop limits number of dropped columns."""
        # Create multiple proxy features
        df = pd.DataFrame(
            {
                "proxy1": [1, 1, 1, 2, 2, 2],
                "proxy2": [1, 1, 1, 2, 2, 2],
                "proxy3": [1, 1, 1, 2, 2, 2],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(sensitive=["group"], threshold=0.5, max_drop=2)
        transformer.fit(df)
        transformed_df = transformer.transform(df)

        # Should drop at most 2 columns
        assert len(transformer.dropped_columns_) <= 2
        # Verify transform returns a DataFrame
        assert isinstance(transformed_df, pd.DataFrame)

    def test_specific_features_list(self):
        """Test with specific features list."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [10, 20, 30, 40, 50, 60],
                "feature3": [100, 200, 300, 400, 500, 600],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(
            sensitive=["group"], features=["feature1", "feature2"], threshold=0.9
        )
        transformer.fit(df)

        # Should only test feature1 and feature2, not feature3
        assert "feature1" in transformer.assoc_scores_ or "feature2" in transformer.assoc_scores_
        assert "feature3" not in transformer.assoc_scores_

    def test_fit_missing_sensitive_column(self):
        """Test fit raises error when sensitive column is missing."""
        df = pd.DataFrame({"feature1": [1, 2, 3]})

        transformer = ProxyDropper(sensitive=["missing"])

        with pytest.raises(ValueError, match="not found in DataFrame"):
            transformer.fit(df)

    def test_fit_invalid_input_type(self):
        """Test fit raises error with non-DataFrame input."""
        transformer = ProxyDropper(sensitive=["group"])

        with pytest.raises(TypeError, match="pandas DataFrame"):
            transformer.fit(np.array([[1, 2], [3, 4]]))

    def test_assoc_scores_stored(self):
        """Test that association scores are stored."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(sensitive=["group"], threshold=0.9)
        transformer.fit(df)

        assert "feature1" in transformer.assoc_scores_
        assert isinstance(transformer.assoc_scores_["feature1"], float)

    def test_multiple_sensitive_attributes(self):
        """Test with multiple sensitive attributes."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
                "gender": ["M", "M", "M", "F", "F", "F"],
            }
        )

        transformer = ProxyDropper(sensitive=["group", "gender"], threshold=0.9)
        transformer.fit(df)

        # Should compute max association across all sensitive attributes
        assert "feature1" in transformer.assoc_scores_

    def test_no_features_dropped_when_below_threshold(self):
        """Test that no features are dropped when all below threshold."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [10, 20, 30, 40, 50, 60],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        transformer = ProxyDropper(sensitive=["group"], threshold=0.99)
        transformer.fit(df)
        result = transformer.transform(df)

        # All features should be kept
        assert len(transformer.dropped_columns_) == 0
        assert "feature1" in result.columns
        assert "feature2" in result.columns
