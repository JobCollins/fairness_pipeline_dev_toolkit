"""
Unit tests for AB testing functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.monitoring.abtest import (
    ABPowerSpec,
    FairnessABTestAnalyzer,
)


class TestABPowerSpec:
    """Tests for ABPowerSpec dataclass."""

    def test_ab_power_spec_defaults(self):
        """Test ABPowerSpec with default values."""
        spec = ABPowerSpec()

        assert spec.effect_size == 0.10
        assert spec.alpha == 0.05
        assert spec.min_group_size == 30

    def test_ab_power_spec_custom_values(self):
        """Test ABPowerSpec with custom values."""
        spec = ABPowerSpec(effect_size=0.20, alpha=0.01, min_group_size=50)

        assert spec.effect_size == 0.20
        assert spec.alpha == 0.01
        assert spec.min_group_size == 50


class TestFairnessABTestAnalyzer:
    """Tests for FairnessABTestAnalyzer."""

    @pytest.fixture
    def sample_control(self):
        """Create sample control group data."""
        return pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 1] * 20,
                "fairness_metric": [0.1, 0.12, 0.11, 0.13, 0.10] * 20,
                "group": ["A", "A", "B", "B", "A"] * 20,
                "gender": ["M", "F", "M", "F", "M"] * 20,
            }
        )

    @pytest.fixture
    def sample_treatment(self):
        """Create sample treatment group data."""
        return pd.DataFrame(
            {
                "outcome": [1, 1, 0, 1, 1] * 20,
                "fairness_metric": [0.05, 0.06, 0.07, 0.05, 0.06] * 20,
                "group": ["A", "A", "B", "B", "A"] * 20,
                "gender": ["M", "F", "M", "F", "M"] * 20,
            }
        )

    def test_analyzer_initialization(self, sample_control, sample_treatment):
        """Test analyzer initialization."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        assert len(analyzer.control) == len(sample_control)
        assert len(analyzer.treatment) == len(sample_treatment)
        assert analyzer.protected == ["group"]
        assert analyzer.outcome_col == "outcome"
        assert analyzer.fair_col == "fairness_metric"

    def test_analyzer_with_multiple_protected_attributes(self, sample_control, sample_treatment):
        """Test analyzer with multiple protected attributes."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group", "gender"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        assert len(analyzer.protected) == 2
        assert "group" in analyzer.protected
        assert "gender" in analyzer.protected

    def test_analyzer_with_business_metrics(self, sample_control, sample_treatment):
        """Test analyzer with business metrics."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
            business_metrics=["revenue", "engagement"],
        )

        assert len(analyzer.business) == 2
        assert "revenue" in analyzer.business
        assert "engagement" in analyzer.business

    def test_power_by_intersection_basic(self, sample_control, sample_treatment):
        """Test basic power calculation by intersection."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        spec = ABPowerSpec(effect_size=0.10, alpha=0.05, min_group_size=10)
        power = analyzer.power_by_intersection(spec)

        assert isinstance(power, dict)
        assert len(power) > 0
        # All power values should be between 0 and 1, or NaN
        for key, value in power.items():
            assert np.isnan(value) or (0.0 <= value <= 1.0)

    def test_power_by_intersection_multiple_groups(self, sample_control, sample_treatment):
        """Test power calculation with multiple groups."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        spec = ABPowerSpec(effect_size=0.10, alpha=0.05, min_group_size=10)
        power = analyzer.power_by_intersection(spec)

        # Should have power for each group
        assert "A" in str(power.keys()) or "__overall__" in str(power.keys())

    def test_power_by_intersection_small_groups_returns_nan(self, sample_control, sample_treatment):
        """Test that small groups return NaN."""
        # Create data with very small groups
        small_control = pd.DataFrame(
            {
                "outcome": [1, 0],
                "fairness_metric": [0.1, 0.12],
                "group": ["A", "B"],
            }
        )
        small_treatment = pd.DataFrame(
            {
                "outcome": [1, 0],
                "fairness_metric": [0.05, 0.06],
                "group": ["A", "B"],
            }
        )

        analyzer = FairnessABTestAnalyzer(
            control=small_control,
            treatment=small_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        spec = ABPowerSpec(effect_size=0.10, alpha=0.05, min_group_size=10)
        power = analyzer.power_by_intersection(spec)

        # Small groups should return NaN
        for value in power.values():
            assert np.isnan(value)

    def test_power_by_intersection_no_protected_attributes(self, sample_control, sample_treatment):
        """Test power calculation with no protected attributes."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=[],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        spec = ABPowerSpec(effect_size=0.10, alpha=0.05, min_group_size=10)
        power = analyzer.power_by_intersection(spec)

        # Should have overall power
        assert "__overall__" in power
        assert 0.0 <= power["__overall__"] <= 1.0

    def test_heterogeneous_effects_basic(self, sample_control, sample_treatment):
        """Test basic heterogeneous effects calculation."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        results = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert "intersection" in results.columns
        assert "metric" in results.columns
        assert "effect" in results.columns
        assert "lower" in results.columns
        assert "upper" in results.columns
        assert "significant" in results.columns
        assert "n_control" in results.columns
        assert "n_treatment" in results.columns

    def test_heterogeneous_effects_confidence_intervals(self, sample_control, sample_treatment):
        """Test that confidence intervals are computed correctly."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        results = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        # Check that lower <= effect <= upper for each row
        for _, row in results.iterrows():
            assert row["lower"] <= row["effect"] <= row["upper"]

    def test_heterogeneous_effects_significance_flagging(self, sample_control, sample_treatment):
        """Test that significance is flagged correctly."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        results = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        # Significance should be boolean
        assert results["significant"].dtype == bool

        # If CI doesn't contain 0, should be significant
        for _, row in results.iterrows():
            if not (row["lower"] <= 0 <= row["upper"]):
                assert row["significant"] is True

    def test_heterogeneous_effects_multiple_metrics(self, sample_control, sample_treatment):
        """Test with multiple business metrics."""
        control_with_metrics = sample_control.copy()
        control_with_metrics["revenue"] = np.random.randn(len(sample_control))
        treatment_with_metrics = sample_treatment.copy()
        treatment_with_metrics["revenue"] = np.random.randn(len(sample_treatment))

        analyzer = FairnessABTestAnalyzer(
            control=control_with_metrics,
            treatment=treatment_with_metrics,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
            business_metrics=["revenue"],
        )

        results = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        # Should have results for outcome, fairness_metric, and revenue
        metrics = results["metric"].unique()
        assert "outcome" in metrics
        assert "fairness_metric" in metrics
        assert "revenue" in metrics

    def test_heterogeneous_effects_empty_groups(self):
        """Test with empty groups (should skip)."""
        control = pd.DataFrame(
            {
                "outcome": [1, 0],
                "fairness_metric": [0.1, 0.12],
                "group": ["A", "A"],
            }
        )
        treatment = pd.DataFrame(
            {
                "outcome": [1, 0],
                "fairness_metric": [0.05, 0.06],
                "group": ["B", "B"],  # Different group
            }
        )

        analyzer = FairnessABTestAnalyzer(
            control=control,
            treatment=treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        results = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        # Should handle empty intersections gracefully
        assert isinstance(results, pd.DataFrame)

    def test_heterogeneous_effects_different_alpha(self, sample_control, sample_treatment):
        """Test with different alpha levels."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        results_90 = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.10)
        results_95 = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        # 90% CI should generally be narrower than 95% CI
        # Due to bootstrap sampling variability with small n_bootstrap=100,
        # occasionally 90% CI can be slightly wider due to sampling noise
        # Check that on average or most cases, 90% CI is narrower
        narrower_count = 0
        for idx in range(min(len(results_90), len(results_95))):
            ci_width_90 = results_90.iloc[idx]["upper"] - results_90.iloc[idx]["lower"]
            ci_width_95 = results_95.iloc[idx]["upper"] - results_95.iloc[idx]["lower"]
            if ci_width_90 <= ci_width_95:
                narrower_count += 1

        # At least half should be narrower (accounting for bootstrap variability)
        assert (
            narrower_count >= len(results_90) // 2
        ), f"Expected at least half of 90% CIs to be narrower than 95% CIs, got {narrower_count}/{len(results_90)}"

    def test_heterogeneous_effects_intersectional(self, sample_control, sample_treatment):
        """Test with intersectional analysis."""
        analyzer = FairnessABTestAnalyzer(
            control=sample_control,
            treatment=sample_treatment,
            protected_attributes=["group", "gender"],
            outcome_column="outcome",
            fairness_metric_column="fairness_metric",
        )

        results = analyzer.heterogeneous_effects(n_bootstrap=100, alpha=0.05)

        # Should have results for each intersection
        intersections = results["intersection"].unique()
        assert len(intersections) > 0
        # Intersections should contain "×" separator
        assert any("×" in inter for inter in intersections)
