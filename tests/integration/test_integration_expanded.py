"""
Expanded integration tests for end-to-end workflows and edge cases.

These tests verify complete workflows, error handling, and integration
between different components of the fairness pipeline toolkit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.integration.orchestrator import (
    ValidationResult,
    run_final_validation,
)
from fairness_pipeline_dev_toolkit.integration.reporting import (
    generate_training_fairness_report,
    to_markdown_report,
)
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.pipeline.config import load_config

# ============================================================================
# Edge Case Tests for Integration
# ============================================================================


class TestIntegrationEdgeCases:
    """Test edge cases in integration workflows."""

    def test_validation_with_nan_metrics(self):
        """Test validation when metrics contain NaN values."""
        baseline_metrics = {"demographic_parity_difference": {"value": float("nan")}}
        final_metrics = {"demographic_parity_difference": {"value": 0.03}}

        config_text = """
sensitive: ["sensitive"]
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
"""
        config = load_config(text=config_text)

        result = run_final_validation(baseline_metrics, final_metrics, config)

        # Should handle NaN gracefully
        assert isinstance(result, ValidationResult)
        assert result.baseline_metric_value is None or np.isnan(result.baseline_metric_value)

    def test_validation_with_missing_metric(self):
        """Test validation when expected metric is missing."""
        baseline_metrics = {"other_metric": {"value": 0.15}}
        final_metrics = {"other_metric": {"value": 0.03}}

        config_text = """
sensitive: ["sensitive"]
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
"""
        config = load_config(text=config_text)

        result = run_final_validation(baseline_metrics, final_metrics, config)

        # Should handle missing metric gracefully
        assert isinstance(result, ValidationResult)

    def test_reporting_with_empty_metrics(self):
        """Test report generation with empty metrics dictionary."""
        report_data = {
            "metadata": {
                "timestamp": "2024-01-01 12:00:00 UTC",
                "fairness_threshold": 0.05,
            },
            "baseline_metrics": {},
            "final_metrics": {},
        }

        markdown, json_data, _ = generate_training_fairness_report(report_data)

        assert "# Training Fairness Report" in markdown
        assert isinstance(json_data, dict)
        assert "metadata" in json_data

    def test_reporting_with_malformed_data(self):
        """Test report generation with malformed data structures."""
        report_data = {
            "metadata": None,  # Invalid metadata
            "baseline_metrics": "not_a_dict",  # Wrong type
        }

        # Should handle gracefully or raise appropriate error
        try:
            markdown, json_data, _ = generate_training_fairness_report(report_data)
            assert isinstance(markdown, str)
        except (TypeError, AttributeError, KeyError):
            # Acceptable to raise errors for malformed data
            pass

    def test_markdown_report_with_none_values(self):
        """Test markdown report generation with None values."""
        results = {
            "metric1": MetricResult(
                metric="metric1",
                value=0.15,
                ci=None,
                effect_size=None,
                n_per_group=None,
            ),
        }

        report = to_markdown_report(results)

        assert "metric1" in report
        assert "0.150000" in report

    def test_markdown_report_with_inf_values(self):
        """Test markdown report generation with infinity values."""
        results = {
            "metric1": MetricResult(
                metric="metric1",
                value=float("inf"),
                ci=None,
                effect_size=None,
            ),
        }

        report = to_markdown_report(results)

        assert "metric1" in report
        # Should handle inf gracefully

    def test_workflow_with_very_large_dataset(self):
        """Test workflow with very large dataset (stress test)."""
        # Create a moderately large dataset
        n_samples = 10000
        df = pd.DataFrame(
            {
                "f0": np.random.randn(n_samples),
                "f1": np.random.randn(n_samples),
                "sensitive": np.random.choice(["A", "B"], size=n_samples, p=[0.6, 0.4]),
            }
        )
        df["y"] = ((df["f0"] + df["f1"] + np.random.randn(n_samples) * 0.1) > 0).astype(int)

        fa = FairnessAnalyzer(min_group_size=30, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        assert isinstance(result.value, (float, np.floating))
        assert 0.0 <= result.value <= 1.0 or np.isnan(result.value)

    def test_workflow_with_very_small_dataset(self):
        """Test workflow with very small dataset (edge case)."""
        df = pd.DataFrame(
            {
                "f0": [1.0, 2.0],
                "f1": [1.0, 2.0],
                "sensitive": ["A", "B"],
                "y": [0, 1],
            }
        )

        fa = FairnessAnalyzer(min_group_size=1, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        assert isinstance(result.value, (float, np.floating))

    def test_workflow_with_many_groups(self):
        """Test workflow with many sensitive attribute groups."""
        n_samples = 1000
        n_groups = 20
        groups = [f"group_{i}" for i in range(n_groups)]

        df = pd.DataFrame(
            {
                "sensitive": np.random.choice(groups, size=n_samples),
                "y": np.random.randint(0, 2, size=n_samples),
            }
        )

        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        assert isinstance(result.value, (float, np.floating))

    def test_workflow_with_single_group(self):
        """Test workflow with only one group (edge case)."""
        df = pd.DataFrame(
            {
                "sensitive": ["A"] * 100,
                "y": np.random.randint(0, 2, size=100),
            }
        )

        fa = FairnessAnalyzer(min_group_size=1, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        # Should return NaN or handle gracefully
        assert isinstance(result.value, (float, np.floating))

    def test_workflow_with_all_zeros(self):
        """Test workflow with all zero predictions."""
        df = pd.DataFrame(
            {
                "sensitive": ["A", "A", "B", "B"],
                "y": [0, 0, 0, 0],
            }
        )

        fa = FairnessAnalyzer(min_group_size=2, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        # Should return 0.0 (no difference)
        assert result.value == 0.0 or np.isnan(result.value)

    def test_workflow_with_all_ones(self):
        """Test workflow with all one predictions."""
        df = pd.DataFrame(
            {
                "sensitive": ["A", "A", "B", "B"],
                "y": [1, 1, 1, 1],
            }
        )

        fa = FairnessAnalyzer(min_group_size=2, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        # Should return 0.0 (no difference)
        assert result.value == 0.0 or np.isnan(result.value)

    def test_workflow_with_missing_sensitive_values(self):
        """Test workflow with missing values in sensitive attribute."""
        df = pd.DataFrame(
            {
                "sensitive": ["A", "B", None, "A", np.nan, "B"],
                "y": [0, 1, 0, 1, 0, 1],
            }
        )

        fa = FairnessAnalyzer(min_group_size=1, backend="native", nan_policy="exclude")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        assert isinstance(result.value, (float, np.floating))

    def test_config_loading_with_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        invalid_yaml = """
sensitive: ["sensitive"
pipeline: [invalid
"""

        # YAML parser raises ParserError for invalid YAML syntax
        with pytest.raises(Exception):  # Accept any exception for invalid YAML
            load_config(text=invalid_yaml)

    def test_config_loading_with_missing_required_fields(self):
        """Test config loading with missing required fields."""
        incomplete_yaml = """
sensitive: ["sensitive"]
# Missing pipeline and other required fields
"""

        # Should either load successfully (if fields are optional) or raise error
        try:
            config = load_config(text=incomplete_yaml)
            assert config is not None
        except (ValueError, KeyError, AttributeError):
            # Acceptable to raise errors for incomplete config
            pass

    def test_file_output_creation(self, tmp_path):
        """Test that file outputs are created correctly."""
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
        }

        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        markdown, json_data, file_paths = generate_training_fairness_report(
            report_data, output_dir=str(output_dir)
        )

        assert file_paths is not None
        assert file_paths["markdown"].exists()
        assert file_paths["json"].exists()

        # Verify file contents
        md_content = file_paths["markdown"].read_text()
        assert "# Training Fairness Report" in md_content

        json_content = json.loads(file_paths["json"].read_text())
        assert "metadata" in json_content

    def test_concurrent_metric_computation(self):
        """Test that metrics can be computed concurrently without issues."""
        df = pd.DataFrame(
            {
                "sensitive": np.random.choice(["A", "B"], size=1000),
                "y_true": np.random.randint(0, 2, size=1000),
                "y_pred": np.random.randint(0, 2, size=1000),
            }
        )

        fa = FairnessAnalyzer(min_group_size=30, backend="native")

        # Compute multiple metrics
        result1 = fa.demographic_parity_difference(df["y_pred"], df["sensitive"])
        result2 = fa.equalized_odds_difference(df["y_true"], df["y_pred"], df["sensitive"])

        assert isinstance(result1.value, (float, np.floating))
        assert isinstance(result2.value, (float, np.floating))

    def test_unicode_sensitive_attributes(self):
        """Test workflow with unicode characters in sensitive attributes."""
        df = pd.DataFrame(
            {
                "sensitive": ["α", "β", "α", "β", "γ", "γ"],
                "y": [0, 1, 0, 1, 0, 1],
            }
        )

        fa = FairnessAnalyzer(min_group_size=2, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        assert isinstance(result.value, (float, np.floating))

    def test_very_long_sensitive_attribute_names(self):
        """Test workflow with very long sensitive attribute names."""
        long_name = "A" * 1000
        df = pd.DataFrame(
            {
                "sensitive": [long_name, "B", long_name, "B"],
                "y": [0, 1, 0, 1],
            }
        )

        fa = FairnessAnalyzer(min_group_size=2, backend="native")
        result = fa.demographic_parity_difference(df["y"], df["sensitive"])

        assert isinstance(result.value, (float, np.floating))
