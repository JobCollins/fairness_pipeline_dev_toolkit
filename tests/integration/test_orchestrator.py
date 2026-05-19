"""
Tests for the integrated workflow orchestrator.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fairness_pipeline_dev_toolkit.integration.orchestrator import (
    ValidationResult,
    WorkflowResult,
    _MitigationSampleWeightMixer,
    _resolve_feature_columns,
    run_baseline_measurement,
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


def test_run_baseline_measurement_returns_baseline_metrics(sample_data):
    """Test that run_baseline_measurement returns baseline_metrics when config has training and fairness_metric."""
    config_text = """
sensitive: ["sensitive"]
fairness_metric: "demographic_parity_difference"
training:
  target_column: "y"
  method: "reductions"
  params: {}
"""
    config = load_config(text=config_text)
    result = run_baseline_measurement(
        sample_data, config, min_group_size=20, train_size=0.8, random_state=42
    )
    assert "sensitive_attribute" in result
    assert "data_shape" in result
    assert "min_group_size" in result
    assert "baseline_metrics" in result
    assert "demographic_parity_difference" in result["baseline_metrics"]
    metric = result["baseline_metrics"]["demographic_parity_difference"]
    assert hasattr(metric, "value")
    assert isinstance(metric.value, (int, float))


def test_run_baseline_measurement_metadata_only(sample_data):
    """Test that run_baseline_measurement returns only metadata when no training/fairness_metric."""
    config_text = """
sensitive: ["sensitive"]
"""
    config = load_config(text=config_text)
    result = run_baseline_measurement(sample_data, config, min_group_size=20)
    assert "sensitive_attribute" in result
    assert "data_shape" in result
    assert "min_group_size" in result
    assert "baseline_metrics" not in result or result.get("baseline_metrics") == {}


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
def test_execute_workflow_reproducible_same_seed(sample_data, sample_config):
    """Same random_state yields identical predictions and validation metrics."""
    kwargs = dict(
        config=sample_config,
        df=sample_data,
        min_group_size=20,
        train_size=0.8,
        random_state=0,
    )
    result_a = execute_workflow(**kwargs)
    result_b = execute_workflow(**kwargs)

    assert np.array_equal(result_a.predictions, result_b.predictions)
    assert result_a.y_test is not None and result_b.y_test is not None
    assert np.array_equal(result_a.y_test, result_b.y_test)
    np.testing.assert_allclose(
        result_a.validation_result.final_metric_value,
        result_b.validation_result.final_metric_value,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result_a.validation_result.baseline_metric_value,
        result_b.validation_result.baseline_metric_value,
        equal_nan=True,
    )


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_execute_workflow_different_seed_differs(sample_data, sample_config):
    """Different random_state values should produce different test partitions."""
    common = dict(
        config=sample_config,
        df=sample_data,
        min_group_size=20,
        train_size=0.8,
    )
    result_0 = execute_workflow(**common, random_state=0)
    result_1 = execute_workflow(**common, random_state=1)

    assert not np.array_equal(result_0.predictions, result_1.predictions)


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_workflow_single_train_test_split(sample_data, sample_config):
    """execute_workflow should call train_test_split exactly once."""
    import fairness_pipeline_dev_toolkit.integration.orchestrator as orch

    original_split = orch.train_test_split
    call_count = 0

    def counting_split(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_split(*args, **kwargs)

    with patch.object(orch, "train_test_split", side_effect=counting_split):
        execute_workflow(
            config=sample_config,
            df=sample_data,
            min_group_size=20,
            train_size=0.8,
            random_state=42,
        )

    assert call_count == 1


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


def test_resolve_feature_columns_missing_raises(sample_data):
    """Explicit features list must exist in the dataframe."""
    with pytest.raises(ValueError, match="not found"):
        _resolve_feature_columns(
            sample_data,
            target_col="y",
            sensitive=["sensitive"],
            features=["f0", "nonexistent"],
        )


def test_resolve_feature_columns_auto_select_numeric(sample_data):
    """Auto mode returns numeric columns excluding target and sensitive."""
    cols = _resolve_feature_columns(
        sample_data,
        target_col="y",
        sensitive=["sensitive"],
        features=None,
        log_auto_select_warning=False,
    )
    assert set(cols) == {"f0", "f1", "f2"}


def test_mitigation_sample_weight_mixer_multiplies_weights():
    """Fairlearn passes sample_weight into the base estimator; mixer multiplies by external weights."""
    np.random.seed(0)
    n = 40
    X = np.random.randn(n, 2)
    y = (X[:, 0] + np.random.randn(n) * 0.2 > 0).astype(int)
    ext = np.ones(n)
    ext[::2] = 2.0
    inner = np.ones(n) * 3.0
    mixed = _MitigationSampleWeightMixer(LogisticRegression(max_iter=500), ext)
    mixed.fit(X, y, sample_weight=inner)
    ref = LogisticRegression(max_iter=500).fit(X, y, sample_weight=ext * inner)
    assert np.array_equal(mixed.predict(X), ref.predict(X))


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_run_transform_explicit_features_excludes_non_numeric_string_column():
    """String columns are ignored for modeling when explicit numeric `features` is set."""
    np.random.seed(1)
    n = 120
    df = pd.DataFrame(
        {
            "f0": np.random.randn(n),
            "f1": np.random.randn(n),
            "notes": ["ok"] * n,
            "sensitive": np.random.choice(["A", "B"], size=n),
            "y": np.random.randint(0, 2, size=n),
        }
    )
    config_text = """
sensitive: ["sensitive"]
features: ["f0", "f1"]
pipeline: []
training:
  method: "reductions"
  target_column: "y"
  params:
    constraint: "demographic_parity"
    eps: 0.05
    T: 5
"""
    config = load_config(text=config_text)
    model, transformed_df, y_full, y_test, predictions = run_transform_and_train(
        df, config, train_size=0.8, random_state=0
    )
    assert model is not None
    assert "notes" not in transformed_df.columns
    assert list(transformed_df.columns) == ["f0", "f1"]
    assert len(predictions) == len(y_test)


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_instance_reweighting_sample_weight_propagates_with_sensitive_in_pipeline_frame():
    """Sensitive attributes in the pipeline frame yield non-trivial instance weights."""
    from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
        apply_pipeline,
        build_pipeline,
    )

    np.random.seed(2)
    n = 300
    df = pd.DataFrame(
        {
            "f0": np.random.randn(n),
            "f1": np.random.randn(n),
            "sensitive": ["A"] * 250 + ["B"] * 50,
            "y": np.random.randint(0, 2, size=n),
        }
    )
    config_text = """
sensitive: ["sensitive"]
features: ["f0", "f1"]
benchmarks:
  sensitive:
    "A": 0.5
    "B": 0.5
pipeline:
  - name: rw
    transformer: "InstanceReweighting"
    params: {}
training:
  method: "reductions"
  target_column: "y"
  params:
    constraint: "demographic_parity"
    eps: 0.05
    T: 3
"""
    config = load_config(text=config_text)
    pipe = build_pipeline(config)
    rng = np.random.RandomState(42)
    train_idx = rng.choice(df.index, size=220, replace=False)
    X_train = df.loc[train_idx, ["f0", "f1"]].copy()
    X_train["sensitive"] = df.loc[train_idx, "sensitive"].values
    result = apply_pipeline(pipe, X_train)
    assert result.metadata is not None
    sw = np.asarray(result.metadata["sample_weight"], dtype=float)
    assert sw.shape[0] == len(X_train)
    assert np.ptp(sw) > 1e-6


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_reductions_predictions_differ_with_instance_reweighting_weights():
    """Mitigation weights must change training relative to uniform weights (regression guard)."""
    np.random.seed(3)
    n = 400
    df = pd.DataFrame(
        {
            "f0": np.random.randn(n),
            "f1": np.random.randn(n),
            "sensitive": ["A"] * 320 + ["B"] * 80,
            "y": ((np.random.randn(n) + (np.arange(n) % 80 == 0).astype(float) * 0.5) > 0).astype(
                int
            ),
        }
    )
    base_cfg = """
sensitive: ["sensitive"]
features: ["f0", "f1"]
benchmarks:
  sensitive:
    "A": 0.5
    "B": 0.5
pipeline:
  - name: rw
    transformer: "InstanceReweighting"
    params: {}
training:
  method: "reductions"
  target_column: "y"
  params:
    constraint: "demographic_parity"
    eps: 0.05
    T: 8
fairness_metric: "demographic_parity_difference"
"""
    cfg_weighted = load_config(text=base_cfg)
    _, _, _, y_test_w, pred_w = run_transform_and_train(
        df, cfg_weighted, train_size=0.75, random_state=7
    )

    cfg_plain = load_config(
        text=base_cfg.replace(
            'pipeline:\n  - name: rw\n    transformer: "InstanceReweighting"\n    params: {}\n',
            "pipeline: []\n",
        )
    )
    _, _, _, y_test_u, pred_u = run_transform_and_train(
        df, cfg_plain, train_size=0.75, random_state=7
    )

    assert np.array_equal(y_test_w, y_test_u)
    assert not np.array_equal(pred_w, pred_u)


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
