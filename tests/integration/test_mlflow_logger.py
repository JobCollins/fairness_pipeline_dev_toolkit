"""
Tests for MLflow workflow logging.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.integration.mlflow_logger import log_workflow_results


# Define minimal dataclasses for testing (avoid importing orchestrator which requires training deps)
@dataclass
class ValidationResult:
    passed: bool
    baseline_metric_value: float
    final_metric_value: float
    threshold: float | None
    improvement: float
    message: str


@dataclass
class WorkflowResult:
    baseline_metrics: dict
    final_metrics: dict
    validation_result: ValidationResult
    model: any
    transformed_df: pd.DataFrame
    predictions: np.ndarray
    y_test: np.ndarray | None = None
    artifacts: dict = field(default_factory=dict)


@pytest.fixture
def mock_workflow_result():
    """Create a mock workflow result for testing."""
    # Use dict format instead of Result object to avoid import issues
    baseline_metric = {"value": 0.15, "ci": None, "effect_size": None, "n_per_group": None}
    final_metric = {"value": 0.05, "ci": None, "effect_size": None, "n_per_group": None}

    return WorkflowResult(
        baseline_metrics={"demographic_parity_difference": baseline_metric},
        final_metrics={"demographic_parity_difference": final_metric},
        validation_result=ValidationResult(
            passed=True,
            baseline_metric_value=0.15,
            final_metric_value=0.05,
            threshold=0.10,
            improvement=0.10,
            message="Validation PASSED",
        ),
        model=MagicMock(spec=["predict"]),  # Mock sklearn model
        transformed_df=pd.DataFrame({"col1": [1, 2, 3]}),
        predictions=np.array([0, 1, 0]),
        y_test=np.array([0, 1, 0]),
        artifacts={"test": "data"},
    )


@pytest.fixture
def mock_pytorch_model():
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.state_dict.return_value = {"layer1.weight": np.array([1.0, 2.0])}
    return model


@patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger._is_mlflow_available")
def test_log_workflow_results_no_mlflow(mock_available, mock_workflow_result, tmp_path):
    """Test that logging gracefully handles MLflow not being available."""
    mock_available.return_value = False

    result = log_workflow_results(
        mock_workflow_result,
        config_path=str(tmp_path / "config.yml"),
        experiment_name="test",
    )

    assert result is False


@patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger._is_mlflow_available")
def test_log_workflow_results_with_mlflow(mock_available, mock_workflow_result, tmp_path):
    """Test logging workflow results to MLflow."""
    mock_available.return_value = True

    # Create a mock mlflow module
    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment.return_value = None
    mock_context = MagicMock()
    mock_mlflow.start_run.return_value = mock_context
    mock_context.__enter__ = MagicMock(return_value=None)
    mock_context.__exit__ = MagicMock(return_value=None)

    # Patch sys.modules to inject our mock
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        # Create a dummy config file
        config_file = tmp_path / "config.yml"
        config_file.write_text("sensitive: ['sensitive']\n")

        result = log_workflow_results(
            mock_workflow_result,
            config_path=str(config_file),
            experiment_name="test_experiment",
            run_name="test_run",
        )

        assert result is True
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once()


@patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger._is_mlflow_available")
def test_log_workflow_results_logs_metrics(mock_available, mock_workflow_result, tmp_path):
    """Test that workflow results log metrics correctly."""
    mock_available.return_value = True

    # Create a mock mlflow module
    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment.return_value = None
    mock_context = MagicMock()
    mock_mlflow.start_run.return_value = mock_context
    mock_context.__enter__ = MagicMock(return_value=None)
    mock_context.__exit__ = MagicMock(return_value=None)

    # Patch sys.modules to inject our mock
    import sys

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        config_file = tmp_path / "config.yml"
        config_file.write_text("sensitive: ['sensitive']\n")

        log_workflow_results(
            mock_workflow_result,
            config_path=str(config_file),
            experiment_name="test",
        )

        # Check that log_metric was called for validation metrics
        metric_calls = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        assert "validation_passed" in metric_calls
        assert "baseline_metric_value" in metric_calls
        assert "final_metric_value" in metric_calls
        assert "improvement" in metric_calls
        assert "accuracy" in metric_calls  # Should log accuracy from y_test and predictions


@patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger._is_mlflow_available")
def test_log_workflow_results_logs_model_sklearn(mock_available, mock_workflow_result, tmp_path):
    """Test that sklearn models are logged as artifacts."""
    mock_available.return_value = True

    # Create a mock mlflow module
    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment.return_value = None
    mock_context = MagicMock()
    mock_mlflow.start_run.return_value = mock_context
    mock_context.__enter__ = MagicMock(return_value=None)
    mock_context.__exit__ = MagicMock(return_value=None)

    mock_joblib = MagicMock()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow, "joblib": mock_joblib}):
        config_file = tmp_path / "config.yml"
        config_file.write_text("sensitive: ['sensitive']\n")

        log_workflow_results(
            mock_workflow_result,
            config_path=str(config_file),
            experiment_name="test",
        )

        # Should attempt to save and log model
        assert mock_joblib.dump.called or mock_mlflow.log_artifact.called


@patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger._is_mlflow_available")
def test_log_workflow_results_logs_config(mock_available, mock_workflow_result, tmp_path):
    """Test that config file is logged as artifact."""
    mock_available.return_value = True

    # Create a mock mlflow module
    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment.return_value = None
    mock_context = MagicMock()
    mock_mlflow.start_run.return_value = mock_context
    mock_context.__enter__ = MagicMock(return_value=None)
    mock_context.__exit__ = MagicMock(return_value=None)

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        config_file = tmp_path / "config.yml"
        config_file.write_text("sensitive: ['sensitive']\n")

        log_workflow_results(
            mock_workflow_result,
            config_path=str(config_file),
            experiment_name="test",
        )

        # Check that config was logged
        artifact_calls = [call[0][0] for call in mock_mlflow.log_artifact.call_args_list]
        assert any("config.yml" in str(call) or "config" in str(call) for call in artifact_calls)
