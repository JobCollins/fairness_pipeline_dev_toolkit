"""
Tests for integration reporting utilities.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.integration.reporting import (
    _analyze_training_convergence,
    _assess_severity,
    _coerce,
    _compute_group_rates,
    _fmt_ci,
    _generate_recommendations,
    _interpret_metric_value,
    _prepare_report_data,
    generate_training_fairness_report,
    to_markdown_report,
)
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.pipeline.detectors.representation import (
    RepresentationResult,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_metric_result():
    """Create a sample MetricResult for testing."""
    return MetricResult(
        metric="demographic_parity_difference",
        value=0.15,
        ci=(0.10, 0.20),
        effect_size=1.5,
        n_per_group={"M": 500, "F": 500},
    )


@pytest.fixture
def sample_metric_dict():
    """Create a sample metric dict for testing."""
    return {
        "value": 0.15,
        "ci": (0.10, 0.20),
        "effect_size": 1.5,
        "n_per_group": {"M": 500, "F": 500},
    }


@pytest.fixture
def minimal_report_data():
    """Create minimal report data for testing."""
    return {
        "metadata": {
            "timestamp": "2024-01-01 12:00:00 UTC",
            "model_name": "test_model",
            "sensitive_attributes": ["race"],
            "fairness_threshold": 0.05,
        }
    }


@pytest.fixture
def full_report_data():
    """Create full report data for testing."""
    return {
        "metadata": {
            "timestamp": "2024-01-01 12:00:00 UTC",
            "model_name": "test_model",
            "sensitive_attributes": ["race"],
            "fairness_threshold": 0.05,
        },
        "data_stage": {
            "representation_bias": {
                "race": RepresentationResult(
                    by_group={"White": 0.6, "Black": 0.3, "Asian": 0.1},
                    benchmarks=None,
                )
            },
            "statistical_disparities": {
                "race": [
                    type(
                        "Disparity",
                        (),
                        {
                            "feature": "income",
                            "test": "t-test",
                            "pvalue": 0.001,
                            "flagged": True,
                        },
                    )()
                ]
            },
            "proxy_variables": {
                "race": [
                    type(
                        "Proxy",
                        (),
                        {
                            "feature": "zipcode",
                            "strength": 0.45,
                            "measure": "correlation",
                            "flagged": True,
                        },
                    )()
                ]
            },
        },
        "baseline_metrics": {
            "demographic_parity": MetricResult(
                metric="demographic_parity_difference",
                value=0.20,
                ci=(0.15, 0.25),
                effect_size=2.0,
                n_per_group={"White": 600, "Black": 300, "Asian": 100},
            )
        },
        "final_metrics": {
            "demographic_parity": MetricResult(
                metric="demographic_parity_difference",
                value=0.03,
                ci=(0.01, 0.05),
                effect_size=1.2,
                n_per_group={"White": 600, "Black": 300, "Asian": 100},
            ),
            "equalized_odds": MetricResult(
                metric="equalized_odds_difference",
                value=0.04,
                ci=(0.02, 0.06),
            ),
        },
        "model_performance": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
        },
        "comparison": {
            "improvement": -0.17,  # Negative means improvement (reduction in unfairness)
            "threshold_status": "pass",
        },
        "mitigation": {
            "lagrangian_training": {
                "convergence": {
                    "converged": True,
                    "violation_trend": "improving",
                    "lambda_trend": "increasing",
                    "initial_violation": 0.20,
                    "final_violation": 0.03,
                    "initial_lambda": 0.01,
                    "final_lambda": 0.15,
                }
            }
        },
    }


# ============================================================================
# Test to_markdown_report()
# ============================================================================


class TestToMarkdownReport:
    """Test suite for to_markdown_report() function."""

    def test_basic_report_generation(self, sample_metric_result):
        """Test basic markdown report generation."""
        results = {
            "demographic_parity_difference": sample_metric_result,
        }
        report = to_markdown_report(results)

        assert "# Fairness Report" in report
        assert "demographic_parity_difference" in report
        assert "0.150000" in report
        assert "[0.1000, 0.2000]" in report
        assert "1.500000" in report

    def test_with_dict_input(self, sample_metric_dict):
        """Test with dictionary input instead of dataclass."""
        results = {
            "demographic_parity_difference": sample_metric_dict,
        }
        report = to_markdown_report(results)

        assert "# Fairness Report" in report
        assert "demographic_parity_difference" in report
        assert "0.150000" in report

    def test_with_ci_and_effect_sizes(self):
        """Test report with CI and effect sizes."""
        results = {
            "metric1": MetricResult(
                metric="metric1",
                value=0.123456,
                ci=(0.100000, 0.150000),
                effect_size=1.234567,
                n_per_group={"A": 100, "B": 200},
            ),
        }
        report = to_markdown_report(results)

        assert "[0.1000, 0.1500]" in report
        assert "1.234567" in report
        assert '"A": 100' in report or "100" in report

    def test_without_ci(self):
        """Test report without CI."""
        results = {
            "metric1": MetricResult(
                metric="metric1",
                value=0.15,
                ci=None,
                effect_size=1.5,
            ),
        }
        report = to_markdown_report(results)

        assert "—" in report  # Should show dash for missing CI

    def test_without_effect_size(self):
        """Test report without effect size."""
        results = {
            "metric1": MetricResult(
                metric="metric1",
                value=0.15,
                ci=(0.10, 0.20),
                effect_size=None,
            ),
        }
        report = to_markdown_report(results)

        assert "—" in report  # Should show dash for missing effect size

    def test_without_n_per_group(self):
        """Test report without n_per_group."""
        results = {
            "metric1": MetricResult(
                metric="metric1",
                value=0.15,
                ci=(0.10, 0.20),
                effect_size=1.5,
                n_per_group=None,
            ),
        }
        report = to_markdown_report(results)

        assert "—" in report  # Should show dash for missing n_per_group

    def test_with_primitive_value(self):
        """Test with primitive value (not dataclass or dict)."""
        results = {
            "metric1": 0.15,
        }
        report = to_markdown_report(results)

        assert "metric1" in report
        assert "0.150000" in report

    def test_multiple_metrics(self):
        """Test report with multiple metrics."""
        results = {
            "dp": MetricResult(
                metric="dp",
                value=0.15,
                ci=(0.10, 0.20),
                effect_size=1.5,
            ),
            "eo": MetricResult(
                metric="eo",
                value=0.08,
                ci=(0.05, 0.11),
                effect_size=1.2,
            ),
        }
        report = to_markdown_report(results)

        assert "dp" in report
        assert "eo" in report
        assert "0.150000" in report
        assert "0.080000" in report

    def test_custom_title(self):
        """Test report with custom title."""
        results = {"metric1": 0.15}
        report = to_markdown_report(results, title="Custom Report Title")

        assert "# Custom Report Title" in report
        assert "# Fairness Report" not in report

    def test_edge_case_nan_value(self):
        """Test with NaN value."""
        results = {
            "metric1": {"value": float("nan")},
        }
        report = to_markdown_report(results)

        assert "metric1" in report
        # NaN should be handled (might show as nan or —)

    def test_edge_case_empty_results(self):
        """Test with empty results dict."""
        results = {}
        report = to_markdown_report(results)

        assert "# Fairness Report" in report
        assert "| Metric | Value |" in report

    def test_edge_case_none_ci_tuple(self):
        """Test with CI tuple containing None values."""
        results = {
            "metric1": {"value": 0.15, "ci": (None, 0.20)},
        }
        report = to_markdown_report(results)

        assert "—" in report  # Should show dash for None CI

    def test_edge_case_zero_value(self):
        """Test with zero value."""
        results = {
            "metric1": MetricResult(metric="metric1", value=0.0),
        }
        report = to_markdown_report(results)

        assert "0.000000" in report

    def test_edge_case_very_small_value(self):
        """Test with very small value."""
        results = {
            "metric1": MetricResult(metric="metric1", value=1e-10),
        }
        report = to_markdown_report(results)

        assert "metric1" in report


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestFmtCi:
    """Test suite for _fmt_ci() function."""

    def test_valid_ci(self):
        """Test formatting valid CI tuple."""
        result = _fmt_ci((0.10, 0.20))
        assert result == "[0.1000, 0.2000]"

    def test_ci_with_none_lower(self):
        """Test CI with None lower bound."""
        result = _fmt_ci((None, 0.20))
        assert result == "—"

    def test_ci_with_none_upper(self):
        """Test CI with None upper bound."""
        result = _fmt_ci((0.10, None))
        assert result == "—"

    def test_ci_with_both_none(self):
        """Test CI with both bounds None."""
        result = _fmt_ci((None, None))
        assert result == "—"

    def test_ci_none(self):
        """Test CI as None."""
        result = _fmt_ci(None)
        assert result == "—"

    def test_ci_empty_tuple(self):
        """Test CI as empty tuple."""
        result = _fmt_ci(())
        # Empty tuple should be falsy, so should return "—"
        assert result == "—"

    def test_ci_precision(self):
        """Test CI formatting precision."""
        result = _fmt_ci((0.123456789, 0.987654321))
        assert result == "[0.1235, 0.9877]"  # 4 decimal places


class TestCoerce:
    """Test suite for _coerce() function."""

    def test_dataclass(self, sample_metric_result):
        """Test coercion of dataclass."""
        result = _coerce(sample_metric_result)
        assert isinstance(result, dict)
        assert "value" in result
        assert result["value"] == 0.15

    def test_dict(self, sample_metric_dict):
        """Test coercion of dict."""
        result = _coerce(sample_metric_dict)
        assert isinstance(result, dict)
        assert result == sample_metric_dict

    def test_primitive_value(self):
        """Test coercion of primitive value."""
        result = _coerce(0.15)
        assert isinstance(result, dict)
        assert result == {"value": 0.15}

    def test_primitive_string(self):
        """Test coercion of string."""
        result = _coerce("test")
        assert isinstance(result, dict)
        assert result == {"value": "test"}

    def test_primitive_int(self):
        """Test coercion of integer."""
        result = _coerce(42)
        assert isinstance(result, dict)
        assert result == {"value": 42}


class TestComputeGroupRates:
    """Test suite for _compute_group_rates() function."""

    def test_basic_computation(self):
        """Test basic group rate computation."""
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        rates = _compute_group_rates(y_pred, sensitive, min_group_size=2)

        assert "A" in rates
        assert "B" in rates
        assert rates["A"] == pytest.approx(2.0 / 3.0)  # 2 out of 3 are 1
        assert rates["B"] == pytest.approx(1.0 / 3.0)  # 1 out of 3 is 1

    def test_min_group_size_filtering(self):
        """Test that groups below min_group_size are filtered."""
        y_pred = np.array([1, 1, 0, 0, 1])
        sensitive = np.array(["A", "A", "A", "B", "B"])

        rates = _compute_group_rates(y_pred, sensitive, min_group_size=3)

        assert "A" in rates  # A has 3 samples
        assert "B" not in rates  # B has only 2 samples

    def test_empty_data(self):
        """Test with empty data."""
        y_pred = np.array([])
        sensitive = np.array([])

        rates = _compute_group_rates(y_pred, sensitive)

        assert rates == {}

    def test_single_group(self):
        """Test with single group."""
        y_pred = np.array([1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A"])

        rates = _compute_group_rates(y_pred, sensitive, min_group_size=2)

        assert "A" in rates
        assert rates["A"] == pytest.approx(0.5)  # 2 out of 4 are 1

    def test_all_zeros(self):
        """Test with all zero predictions."""
        y_pred = np.array([0, 0, 0, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        rates = _compute_group_rates(y_pred, sensitive, min_group_size=2)

        assert rates["A"] == 0.0
        assert rates["B"] == 0.0

    def test_all_ones(self):
        """Test with all one predictions."""
        y_pred = np.array([1, 1, 1, 1])
        sensitive = np.array(["A", "A", "B", "B"])

        rates = _compute_group_rates(y_pred, sensitive, min_group_size=2)

        assert rates["A"] == 1.0
        assert rates["B"] == 1.0


class TestAssessSeverity:
    """Test suite for _assess_severity() function."""

    def test_critical_severity(self):
        """Test critical severity (>0.10)."""
        severity, msg = _assess_severity(0.15, 0.05)
        assert severity == "Critical"
        assert "Immediate action required" in msg

    def test_high_severity_above_threshold(self):
        """Test high severity (above threshold but <=0.10)."""
        severity, msg = _assess_severity(0.08, 0.05)
        assert severity == "High"
        assert "Exceeds threshold" in msg
        assert "60.0%" in msg  # (0.08 - 0.05) / 0.05 * 100 = 60%

    def test_medium_severity(self):
        """Test medium severity (>0.02 but <= threshold)."""
        severity, msg = _assess_severity(0.03, 0.05)
        assert severity == "Medium"
        assert "acceptable range" in msg

    def test_low_severity(self):
        """Test low severity (<=0.02)."""
        severity, msg = _assess_severity(0.01, 0.05)
        assert severity == "Low"
        assert "Meets fairness standards" in msg

    def test_exactly_at_threshold(self):
        """Test value exactly at threshold."""
        severity, msg = _assess_severity(0.05, 0.05)
        assert severity == "Medium"  # > 0.02 but <= threshold

    def test_zero_value(self):
        """Test with zero value."""
        severity, msg = _assess_severity(0.0, 0.05)
        assert severity == "Low"

    def test_very_small_value(self):
        """Test with very small value."""
        severity, msg = _assess_severity(0.001, 0.05)
        assert severity == "Low"


class TestInterpretMetricValue:
    """Test suite for _interpret_metric_value() function."""

    def test_demographic_parity_excellent(self):
        """Test interpretation for excellent DP value."""
        result = _interpret_metric_value(0.01, "demographic_parity_difference", 0.05)
        assert "Demographic Parity Difference" in result
        assert "1.00 percentage points" in result
        assert "excellent" in result.lower()

    def test_demographic_parity_acceptable(self):
        """Test interpretation for acceptable DP value."""
        result = _interpret_metric_value(0.04, "demographic_parity_difference", 0.05)
        assert "Demographic Parity Difference" in result
        assert "acceptable" in result.lower()

    def test_demographic_parity_concerning(self):
        """Test interpretation for concerning DP value."""
        result = _interpret_metric_value(0.08, "demographic_parity_difference", 0.05)
        assert "Demographic Parity Difference" in result
        assert "exceeds the threshold" in result.lower()

    def test_demographic_parity_no_threshold(self):
        """Test interpretation without threshold."""
        result = _interpret_metric_value(0.05, "demographic_parity_difference", None)
        assert "Demographic Parity Difference" in result
        assert "5.00 percentage points" in result

    def test_equalized_odds(self):
        """Test interpretation for equalized odds."""
        result = _interpret_metric_value(0.06, "equalized_odds_difference")
        assert "Equalized Odds Difference" in result
        assert "6.00 percentage point difference" in result

    def test_unknown_metric(self):
        """Test interpretation for unknown metric."""
        result = _interpret_metric_value(0.05, "unknown_metric")
        assert "Metric value: 0.0500" in result


# ============================================================================
# Test _analyze_training_convergence()
# ============================================================================


class TestAnalyzeTrainingConvergence:
    """Test suite for _analyze_training_convergence() function."""

    def test_insufficient_data_empty(self):
        """Test with empty history."""
        result = _analyze_training_convergence([])
        assert result["converged"] is False
        assert result["violation_trend"] == "insufficient_data"
        assert result["lambda_trend"] == "insufficient_data"

    def test_insufficient_data_single_entry(self):
        """Test with single entry in history."""
        history = [{"violation": 0.20, "lambda": 0.01}]
        result = _analyze_training_convergence(history)
        assert result["converged"] is False
        assert result["violation_trend"] == "insufficient_data"

    def test_improving_violations(self):
        """Test with improving violations (decreased by >50%)."""
        history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.08, "lambda": 0.05},  # 0.08 < 0.20 * 0.5
        ]
        result = _analyze_training_convergence(history)
        assert result["violation_trend"] == "improving"
        assert result["initial_violation"] == 0.20
        assert result["final_violation"] == 0.08

    def test_slight_improvement(self):
        """Test with slight improvement (decreased but <50%)."""
        history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.15, "lambda": 0.05},  # 0.15 < 0.20 but > 0.20 * 0.5
        ]
        result = _analyze_training_convergence(history)
        assert result["violation_trend"] == "slight_improvement"

    def test_worsening_violations(self):
        """Test with worsening violations (increased by >50%)."""
        history = [
            {"violation": 0.10, "lambda": 0.01},
            {"violation": 0.20, "lambda": 0.05},  # 0.20 > 0.10 * 1.5
        ]
        result = _analyze_training_convergence(history)
        assert result["violation_trend"] == "worsening"

    def test_stable_violations(self):
        """Test with stable violations."""
        history = [
            {"violation": 0.10, "lambda": 0.01},
            {"violation": 0.12, "lambda": 0.05},  # Between 0.10 and 0.15
        ]
        result = _analyze_training_convergence(history)
        assert result["violation_trend"] == "stable"

    def test_increasing_lambda(self):
        """Test with increasing lambda (>1.5x)."""
        history = [
            {"violation": 0.10, "lambda": 0.01},
            {"violation": 0.08, "lambda": 0.02},  # 0.02 > 0.01 * 1.5
        ]
        result = _analyze_training_convergence(history)
        assert result["lambda_trend"] == "increasing"

    def test_slight_lambda_increase(self):
        """Test with slight lambda increase."""
        history = [
            {"violation": 0.10, "lambda": 0.01},
            {"violation": 0.08, "lambda": 0.015},  # 0.015 > 0.01 but < 0.01 * 1.5
        ]
        result = _analyze_training_convergence(history)
        assert result["lambda_trend"] == "slight_increase"

    def test_stable_lambda(self):
        """Test with stable lambda."""
        history = [
            {"violation": 0.10, "lambda": 0.01},
            {"violation": 0.08, "lambda": 0.01},  # Same lambda
        ]
        result = _analyze_training_convergence(history)
        assert result["lambda_trend"] == "stable"

    def test_converged(self):
        """Test converged scenario (improving violations + increasing lambda)."""
        history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.08, "lambda": 0.02},  # Improving + increasing
        ]
        result = _analyze_training_convergence(history)
        assert result["converged"] is True

    def test_not_converged_worsening(self):
        """Test not converged due to worsening violations."""
        history = [
            {"violation": 0.10, "lambda": 0.01},
            {"violation": 0.20, "lambda": 0.02},  # Worsening
        ]
        result = _analyze_training_convergence(history)
        assert result["converged"] is False

    def test_not_converged_stable_lambda(self):
        """Test not converged due to stable lambda."""
        history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.08, "lambda": 0.01},  # Improving but stable lambda
        ]
        result = _analyze_training_convergence(history)
        assert result["converged"] is False

    def test_multiple_epochs(self):
        """Test with multiple epochs."""
        history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.15, "lambda": 0.02},
            {"violation": 0.10, "lambda": 0.03},
            {"violation": 0.05, "lambda": 0.04},
        ]
        result = _analyze_training_convergence(history)
        assert result["initial_violation"] == 0.20
        assert result["final_violation"] == 0.05
        assert result["initial_lambda"] == 0.01
        assert result["final_lambda"] == 0.04

    def test_missing_violation_key(self):
        """Test with missing violation key (defaults to 0)."""
        history = [
            {"lambda": 0.01},
            {"lambda": 0.02},
        ]
        result = _analyze_training_convergence(history)
        # Should handle missing keys gracefully
        assert "violation_trend" in result

    def test_missing_lambda_key(self):
        """Test with missing lambda key (defaults to 0)."""
        history = [
            {"violation": 0.10},
            {"violation": 0.08},
        ]
        result = _analyze_training_convergence(history)
        # Should handle missing keys gracefully
        assert "lambda_trend" in result


# ============================================================================
# Test _generate_recommendations()
# ============================================================================


class TestGenerateRecommendations:
    """Test suite for _generate_recommendations() function."""

    def test_data_stage_representation_bias(self):
        """Test recommendations for representation bias."""
        report_data = {
            "data_stage": {
                "representation_bias": {
                    "race": RepresentationResult(
                        by_group={"White": 0.7, "Black": 0.2, "Asian": 0.1},
                        benchmarks=None,
                    )
                }
            }
        }
        # Create a mock result with proportions attribute
        result = type("Result", (), {"proportions": {"White": 0.7, "Black": 0.2, "Asian": 0.1}})()
        report_data["data_stage"]["representation_bias"]["race"] = result

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["data_stage"]) > 0
        assert any("balanced data" in rec.lower() for rec in recommendations["data_stage"])

    def test_data_stage_proxy_variables(self):
        """Test recommendations for proxy variables."""
        proxy = type(
            "Proxy",
            (),
            {
                "feature": "zipcode",
                "flagged": True,
                "strength": 0.45,
                "measure": "correlation",
            },
        )()
        report_data = {"data_stage": {"proxy_variables": {"race": [proxy]}}}

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["data_stage"]) > 0
        assert any("zipcode" in rec.lower() for rec in recommendations["data_stage"])

    def test_data_stage_statistical_disparities(self):
        """Test recommendations for statistical disparities."""
        disparity = type(
            "Disparity",
            (),
            {
                "feature": "income",
                "flagged": True,
                "pvalue": 0.001,
                "test": "t-test",
            },
        )()
        report_data = {"data_stage": {"statistical_disparities": {"race": [disparity]}}}

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["data_stage"]) > 0
        assert any("income" in rec.lower() for rec in recommendations["data_stage"])

    def test_training_stage_high_dp(self):
        """Test recommendations for high demographic parity."""
        dp_result = MetricResult(
            metric="demographic_parity_difference",
            value=0.08,  # Above threshold of 0.05
        )
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {"demographic_parity": dp_result},
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["training_stage"]) > 0
        assert any(
            "fairness constraint" in rec.lower() for rec in recommendations["training_stage"]
        )

    def test_training_stage_not_converged_worsening(self):
        """Test recommendations for non-converged training with worsening violations."""
        report_data = {
            "mitigation": {
                "lagrangian_training": {
                    "convergence": {
                        "converged": False,
                        "violation_trend": "worsening",
                    }
                }
            }
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["training_stage"]) > 0
        assert any(
            "violations increased" in rec.lower() for rec in recommendations["training_stage"]
        )

    def test_training_stage_stable_lambda(self):
        """Test recommendations for stable lambda."""
        report_data = {
            "mitigation": {
                "lagrangian_training": {
                    "convergence": {
                        "converged": False,
                        "lambda_trend": "stable",
                    }
                }
            }
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["training_stage"]) > 0
        assert any("lambda_lr" in rec.lower() for rec in recommendations["training_stage"])

    def test_evaluation_stage_ci_includes_zero(self):
        """Test recommendations when CI includes zero."""
        dp_result = MetricResult(
            metric="demographic_parity_difference",
            value=0.03,
            ci=(-0.01, 0.07),  # CI includes zero
        )
        report_data = {
            "final_metrics": {"demographic_parity": dp_result},
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["evaluation_stage"]) > 0
        assert any("sample size" in rec.lower() for rec in recommendations["evaluation_stage"])

    def test_evaluation_stage_high_effect_size(self):
        """Test recommendations for high effect size."""
        dp_result = MetricResult(
            metric="demographic_parity_difference",
            value=0.05,
            effect_size=2.0,  # > 1.5
        )
        report_data = {
            "final_metrics": {"demographic_parity": dp_result},
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["evaluation_stage"]) > 0
        assert any(
            "effect_size" in rec.lower() or "risk ratio" in rec.lower()
            for rec in recommendations["evaluation_stage"]
        )

    def test_deployment_stage_fail(self):
        """Test recommendations for deployment failure."""
        dp_result = MetricResult(
            metric="demographic_parity_difference",
            value=0.08,  # Above threshold
        )
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {"demographic_parity": dp_result},
            "comparison": {"threshold_status": "fail"},
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["deployment_stage"]) > 0
        assert any("do not deploy" in rec.lower() for rec in recommendations["deployment_stage"])

    def test_deployment_stage_pass_close_to_threshold(self):
        """Test recommendations for passing but close to threshold."""
        dp_result = MetricResult(
            metric="demographic_parity_difference",
            value=0.045,  # 0.045 > 0.05 * 0.8 = 0.04
        )
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {"demographic_parity": dp_result},
            "comparison": {"threshold_status": "pass"},
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["deployment_stage"]) > 0
        assert any("monitoring" in rec.lower() for rec in recommendations["deployment_stage"])

    def test_deployment_stage_pass_safe(self):
        """Test recommendations for safe deployment."""
        dp_result = MetricResult(
            metric="demographic_parity_difference",
            value=0.02,  # Well below threshold
        )
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {"demographic_parity": dp_result},
            "comparison": {"threshold_status": "pass"},
        }

        recommendations = _generate_recommendations(report_data)

        assert len(recommendations["deployment_stage"]) > 0
        assert any("safe to deploy" in rec.lower() for rec in recommendations["deployment_stage"])

    def test_no_recommendations_empty_data(self):
        """Test with empty report data."""
        report_data = {}
        recommendations = _generate_recommendations(report_data)

        assert recommendations["data_stage"] == []
        assert recommendations["training_stage"] == []
        assert recommendations["evaluation_stage"] == []
        assert recommendations["deployment_stage"] == []


# ============================================================================
# Test _prepare_report_data()
# ============================================================================


class TestPrepareReportData:
    """Test suite for _prepare_report_data() function."""

    def test_auto_compute_performance(self):
        """Test auto-computation of performance metrics."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])  # Some errors

        report_data = {
            "y_true": y_true,
            "y_pred": y_pred,
        }

        result = _prepare_report_data(report_data, compute_performance=True)

        assert "model_performance" in result
        assert "accuracy" in result["model_performance"]
        assert "precision" in result["model_performance"]
        assert "recall" in result["model_performance"]
        assert "f1" in result["model_performance"]

    def test_auto_compute_performance_disabled(self):
        """Test that performance is not computed when disabled."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])

        report_data = {
            "y_true": y_true,
            "y_pred": y_pred,
        }

        result = _prepare_report_data(report_data, compute_performance=False)

        assert "model_performance" not in result

    def test_auto_compute_performance_already_exists(self):
        """Test that existing performance metrics are not overwritten."""
        report_data = {
            "y_true": np.array([1, 0, 1, 0]),
            "y_pred": np.array([1, 0, 1, 0]),
            "model_performance": {"accuracy": 0.99},
        }

        result = _prepare_report_data(report_data, compute_performance=True)

        assert result["model_performance"]["accuracy"] == 0.99  # Not overwritten

    def test_auto_compute_convergence(self):
        """Test auto-computation of convergence analysis."""
        training_history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.10, "lambda": 0.02},
            {"violation": 0.05, "lambda": 0.03},
        ]

        report_data = {
            "mitigation": {"lagrangian_training": {}},
            "training_history": training_history,
        }

        result = _prepare_report_data(report_data, compute_convergence=True)

        assert "convergence" in result["mitigation"]["lagrangian_training"]
        assert "converged" in result["mitigation"]["lagrangian_training"]["convergence"]

    def test_auto_compute_convergence_from_lagrangian(self):
        """Test convergence computation from lagrangian history."""
        training_history = [
            {"violation": 0.20, "lambda": 0.01},
            {"violation": 0.10, "lambda": 0.02},
        ]

        report_data = {
            "mitigation": {
                "lagrangian_training": {
                    "history": training_history,
                }
            },
        }

        result = _prepare_report_data(report_data, compute_convergence=True)

        assert "convergence" in result["mitigation"]["lagrangian_training"]

    def test_auto_compute_convergence_disabled(self):
        """Test that convergence is not computed when disabled."""
        training_history = [
            {"violation": 0.20, "lambda": 0.01},
        ]

        report_data = {
            "training_history": training_history,
        }

        result = _prepare_report_data(report_data, compute_convergence=False)

        assert "convergence" not in result.get("mitigation", {}).get("lagrangian_training", {})

    def test_auto_compute_convergence_already_exists(self):
        """Test that existing convergence is not overwritten."""
        report_data = {
            "mitigation": {
                "lagrangian_training": {
                    "convergence": {"converged": True},
                    "history": [{"violation": 0.20, "lambda": 0.01}],
                }
            },
        }

        result = _prepare_report_data(report_data, compute_convergence=True)

        # Should not recompute if already exists
        assert result["mitigation"]["lagrangian_training"]["convergence"]["converged"] is True

    def test_auto_compute_weight_stats(self):
        """Test auto-computation of weight statistics."""
        sample_weights = np.array([0.5, 1.0, 1.5, 2.0, 0.8])

        report_data = {
            "mitigation": {"instance_reweighting": {}},
            "sample_weights": sample_weights,
        }

        result = _prepare_report_data(report_data)

        assert "weights_stats" in result["mitigation"]["instance_reweighting"]
        assert "min" in result["mitigation"]["instance_reweighting"]["weights_stats"]
        assert "max" in result["mitigation"]["instance_reweighting"]["weights_stats"]
        assert "mean" in result["mitigation"]["instance_reweighting"]["weights_stats"]

    def test_auto_compute_weight_stats_from_reweighting(self):
        """Test weight stats from reweighting section."""
        sample_weights = np.array([0.5, 1.0, 1.5])

        report_data = {
            "mitigation": {
                "instance_reweighting": {
                    "sample_weights": sample_weights,
                }
            },
        }

        result = _prepare_report_data(report_data)

        assert "weights_stats" in result["mitigation"]["instance_reweighting"]

    def test_auto_add_timestamp(self):
        """Test that timestamp is auto-added if missing."""
        report_data = {
            "metadata": {},
        }

        result = _prepare_report_data(report_data)

        assert "timestamp" in result["metadata"]

    def test_timestamp_not_overwritten(self):
        """Test that existing timestamp is not overwritten."""
        report_data = {
            "metadata": {"timestamp": "2024-01-01 12:00:00 UTC"},
        }

        result = _prepare_report_data(report_data)

        assert result["metadata"]["timestamp"] == "2024-01-01 12:00:00 UTC"

    def test_creates_metadata_if_missing(self):
        """Test that metadata is created if missing."""
        report_data = {}

        result = _prepare_report_data(report_data)

        assert "metadata" in result
        assert "timestamp" in result["metadata"]


# ============================================================================
# Test generate_training_fairness_report()
# ============================================================================


class TestGenerateTrainingFairnessReport:
    """Test suite for generate_training_fairness_report() function."""

    def test_minimal_data(self, minimal_report_data):
        """Test report generation with minimal data."""
        markdown, json_data, file_paths = generate_training_fairness_report(minimal_report_data)

        assert isinstance(markdown, str)
        assert "# Training Fairness Report" in markdown
        assert isinstance(json_data, dict)
        assert file_paths is None  # No output_dir provided

    def test_full_data(self, full_report_data):
        """Test report generation with full data."""
        markdown, json_data, file_paths = generate_training_fairness_report(full_report_data)

        assert isinstance(markdown, str)
        assert "# Training Fairness Report" in markdown
        assert "Executive Summary" in markdown
        assert "Data Quality & Bias Detection" in markdown
        assert "Baseline Fairness Assessment" in markdown
        assert "Mitigation Strategy Applied" in markdown
        assert "Final Fairness Evaluation" in markdown
        assert "Actionable Recommendations" in markdown
        assert "Model Performance Context" in markdown

        assert isinstance(json_data, dict)
        assert "metadata" in json_data
        assert "executive_summary" in json_data

    def test_file_output(self, full_report_data, tmp_path):
        """Test report generation with file output."""
        output_dir = tmp_path / "reports"
        markdown, json_data, file_paths = generate_training_fairness_report(
            full_report_data, output_dir=str(output_dir)
        )

        assert file_paths is not None
        assert "markdown" in file_paths
        assert "json" in file_paths

        # Check files exist
        assert file_paths["markdown"].exists()
        assert file_paths["json"].exists()

        # Check file contents
        md_content = file_paths["markdown"].read_text()
        assert "# Training Fairness Report" in md_content

        json_content = json.loads(file_paths["json"].read_text())
        assert "metadata" in json_content

    def test_json_serialization(self, full_report_data):
        """Test that JSON data is serializable."""
        markdown, json_data, file_paths = generate_training_fairness_report(full_report_data)

        # Should be able to serialize to JSON
        json_str = json.dumps(json_data, default=str)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should be able to deserialize
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_auto_compute_performance(self):
        """Test auto-computation of performance in report."""
        report_data = {
            "metadata": {
                "timestamp": "2024-01-01 12:00:00 UTC",
                "fairness_threshold": 0.05,
            },
            "final_metrics": {
                "demographic_parity": MetricResult(
                    metric="demographic_parity_difference",
                    value=0.03,
                ),
            },
            "comparison": {"threshold_status": "pass"},
            "y_true": np.array([1, 0, 1, 0, 1, 0]),
            "y_pred": np.array([1, 0, 1, 0, 1, 0]),
        }

        markdown, json_data, _ = generate_training_fairness_report(
            report_data, compute_performance=True
        )

        assert "Model Performance Context" in markdown
        assert "model_performance" in json_data
        assert "accuracy" in json_data["model_performance"]

    def test_auto_compute_convergence(self):
        """Test auto-computation of convergence in report."""
        report_data = {
            "metadata": {
                "timestamp": "2024-01-01 12:00:00 UTC",
                "fairness_threshold": 0.05,
            },
            "final_metrics": {
                "demographic_parity": MetricResult(
                    metric="demographic_parity_difference",
                    value=0.03,
                ),
            },
            "comparison": {"threshold_status": "pass"},
            "mitigation": {"lagrangian_training": {}},
            "training_history": [
                {"violation": 0.20, "lambda": 0.01},
                {"violation": 0.10, "lambda": 0.02},
            ],
        }

        markdown, json_data, _ = generate_training_fairness_report(
            report_data, compute_convergence=True
        )

        assert "Lagrangian Training" in markdown
        assert "convergence" in json_data["mitigation"]["lagrangian_training"]

    def test_pass_status(self):
        """Test report with pass status."""
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {
                "demographic_parity": MetricResult(
                    metric="demographic_parity_difference",
                    value=0.03,
                ),
            },
            "comparison": {"threshold_status": "pass"},
        }

        markdown, json_data, _ = generate_training_fairness_report(report_data)

        assert "✅ **PASS**" in markdown
        assert json_data["executive_summary"]["status"] == "pass"

    def test_fail_status(self):
        """Test report with fail status."""
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {
                "demographic_parity": MetricResult(
                    metric="demographic_parity_difference",
                    value=0.08,
                ),
            },
            "comparison": {"threshold_status": "fail"},
        }

        markdown, json_data, _ = generate_training_fairness_report(report_data)

        assert "❌ **FAIL**" in markdown
        assert json_data["executive_summary"]["status"] == "fail"

    def test_warn_status(self):
        """Test report with warn status."""
        report_data = {
            "metadata": {"fairness_threshold": 0.05},
            "final_metrics": {
                "demographic_parity": MetricResult(
                    metric="demographic_parity_difference",
                    value=0.03,
                ),
            },
            "comparison": {"threshold_status": "unknown"},
        }

        markdown, json_data, _ = generate_training_fairness_report(report_data)

        assert "⚠️ **WARN**" in markdown
        assert json_data["executive_summary"]["status"] == "unknown"

    def test_recommendations_included(self, full_report_data):
        """Test that recommendations are included in report."""
        markdown, json_data, _ = generate_training_fairness_report(full_report_data)

        assert "Actionable Recommendations" in markdown
        assert "recommendations" in json_data
        assert "data_stage" in json_data["recommendations"]

    def test_dataclass_serialization(self, full_report_data):
        """Test that dataclasses are properly serialized."""
        markdown, json_data, _ = generate_training_fairness_report(full_report_data)

        # Should be able to serialize without errors
        json_str = json.dumps(json_data, default=str)
        assert isinstance(json_str, str)

    def test_empty_sections_handled(self, minimal_report_data):
        """Test that empty sections are handled gracefully."""
        markdown, json_data, _ = generate_training_fairness_report(minimal_report_data)

        # Should still generate valid report
        assert "# Training Fairness Report" in markdown
        assert isinstance(json_data, dict)
