"""
Tests for the integrated workflow orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.integration.orchestrator import (
    ValidationResult,
    WorkflowResult,
    run_final_validation,
)
from fairness_pipeline_dev_toolkit.pipeline.config import load_config

# Import training-dependent functions only if available
try:
    from fairness_pipeline_dev_toolkit.integration.orchestrator import (
        execute_workflow,
        run_transform_and_train,
    )

    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "f0": np.random.randn(n_samples),
            "f1": np.random.randn(n_samples),
            "f2": np.random.randn(n_samples),
            "sensitive": np.random.choice(["A", "B"], size=n_samples, p=[0.6, 0.4]),
        }
    )

    # Target with some bias
    bias = (df["sensitive"] == "B").astype(int) * 0.3
    df["y"] = ((df["f0"] + df["f1"] + bias + np.random.randn(n_samples) * 0.1) > 0).astype(int)

    return df


@pytest.fixture
def sample_config():
    """Create sample config for testing."""
    config_text = """
sensitive: ["sensitive"]
alpha: 0.05
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
training:
  method: "reductions"
  target_column: "y"
  params:
    constraint: "demographic_parity"
    eps: 0.01
    T: 10
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.10
"""
    return load_config(text=config_text)


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_run_transform_and_train_reductions(sample_data, sample_config):
    """Test transform and train with reductions method."""
    model, transformed_df, y_full, y_test, predictions = run_transform_and_train(
        sample_data, sample_config, train_size=0.8
    )

    assert model is not None
    assert len(transformed_df) == len(sample_data)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_run_final_validation_passed():
    """Test final validation when threshold is met."""
    baseline_metrics = {"demographic_parity_difference": {"value": 0.15}}
    final_metrics = {"demographic_parity_difference": {"value": 0.03}}

    config_text = """
sensitive: ["sensitive"]
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
"""
    config = load_config(text=config_text)

    result = run_final_validation(baseline_metrics, final_metrics, config)

    assert result.passed is True
    assert result.baseline_metric_value == 0.15
    assert result.final_metric_value == 0.03
    assert result.improvement == 0.12  # 0.15 - 0.03


def test_run_final_validation_failed():
    """Test final validation when threshold is not met."""
    baseline_metrics = {"demographic_parity_difference": {"value": 0.15}}
    final_metrics = {"demographic_parity_difference": {"value": 0.08}}

    config_text = """
sensitive: ["sensitive"]
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
"""
    config = load_config(text=config_text)

    result = run_final_validation(baseline_metrics, final_metrics, config)

    assert result.passed is False
    assert result.final_metric_value == 0.08
    assert abs(result.final_metric_value) > config.validation_threshold


def test_run_final_validation_no_threshold():
    """Test final validation when no threshold is specified."""
    baseline_metrics = {"demographic_parity_difference": {"value": 0.15}}
    final_metrics = {"demographic_parity_difference": {"value": 0.03}}

    config_text = """
sensitive: ["sensitive"]
fairness_metric: "demographic_parity_difference"
"""
    config = load_config(text=config_text)

    result = run_final_validation(baseline_metrics, final_metrics, config)

    assert result.passed is True  # No threshold means always pass
    assert result.threshold is None


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_execute_workflow_end_to_end(sample_data, sample_config, tmp_path):
    """Test complete workflow execution."""
    result = execute_workflow(
        config=sample_config,
        df=sample_data,
        output_dir=str(tmp_path),
        min_group_size=20,
        train_size=0.8,
    )

    assert isinstance(result, WorkflowResult)
    assert result.model is not None
    assert len(result.transformed_df) == len(sample_data)
    assert len(result.predictions) > 0
    assert isinstance(result.validation_result, ValidationResult)
    assert "baseline_metrics" in result.artifacts
    assert "final_metrics" in result.artifacts

    # Check artifacts were saved
    assert (tmp_path / "workflow_results.json").exists()
    assert (tmp_path / "transformed_data.csv").exists()


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_execute_workflow_no_output_dir(sample_data, sample_config):
    """Test workflow execution without output directory."""
    result = execute_workflow(
        config=sample_config,
        df=sample_data,
        output_dir=None,
        min_group_size=20,
    )

    assert isinstance(result, WorkflowResult)
    assert result.model is not None


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_execute_workflow_requires_training(sample_data):
    """Test that workflow requires training section in config."""
    config_text = """
sensitive: ["sensitive"]
pipeline: []
"""
    config = load_config(text=config_text)

    from fairness_pipeline_dev_toolkit.exceptions import TrainingError

    with pytest.raises(TrainingError, match="training"):
        execute_workflow(config=config, df=sample_data)


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="PyTorch required for regularized/lagrangian methods",
)
def test_run_transform_and_train_regularized(sample_data):
    """Test transform and train with regularized method (requires PyTorch)."""
    config_text = """
sensitive: ["sensitive"]
pipeline: []
training:
  method: "regularized"
  target_column: "y"
  params:
    eta: 0.5
    epochs: 5
    lr: 0.001
"""
    config = load_config(text=config_text)

    model, transformed_df, y_full, y_test, predictions = run_transform_and_train(
        sample_data, config, train_size=0.8
    )

    assert model is not None
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_workflow_result_dataclass():
    """Test WorkflowResult dataclass structure."""
    result = WorkflowResult(
        baseline_metrics={},
        final_metrics={},
        validation_result=ValidationResult(
            passed=True,
            baseline_metric_value=0.1,
            final_metric_value=0.05,
            threshold=0.1,
            improvement=0.05,
            message="Test",
        ),
        model=None,
        transformed_df=pd.DataFrame(),
        predictions=np.array([0, 1]),
    )

    assert result.validation_result.passed is True
    assert result.validation_result.improvement == 0.05
