"""
Orchestrator for integrated three-step fairness workflow.

Step 1: Baseline Measurement - audit raw data
Step 2: Transform Data + Train Model - apply pipeline transformers, train model
Step 3: Final Validation - compare to baseline, check validation threshold
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fairness_pipeline_dev_toolkit.exceptions import (
    DataValidationError,
    DependencyError,
    TrainingError,
)
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
)
from fairness_pipeline_dev_toolkit.utils.logging import (
    PerformanceLogger,
    get_logger,
    log_metric_computation,
    log_workflow_step,
)

logger = get_logger("integration.orchestrator")

# Training imports are lazy/conditional to avoid requiring optional dependencies
# Imported only when needed in run_transform_and_train()


class _MitigationSampleWeightMixer(BaseEstimator, ClassifierMixin):
    """
    Wraps a classifier so Fairlearn reductions' internal sample_weight is multiplied
    row-wise by external weights (e.g. from InstanceReweighting). Fairlearn's
    ExponentiatedGradient does not accept a top-level ``sample_weight`` kwarg.
    """

    def __init__(self, estimator: Any, external_sample_weight: np.ndarray):
        self.estimator = estimator
        self.external_sample_weight = external_sample_weight

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "_MitigationSampleWeightMixer":
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        ext = np.asarray(self.external_sample_weight, dtype=float).reshape(-1)
        if ext.shape[0] != n:
            raise ValueError(
                f"external_sample_weight length {ext.shape[0]} does not match X rows {n}."
            )
        if sample_weight is not None:
            sw = ext * np.asarray(sample_weight, dtype=float).reshape(-1)
        else:
            sw = ext
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=sw)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)
        raise AttributeError("Underlying estimator has no predict_proba")


def _resolve_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    sensitive: Sequence[str],
    features: Optional[List[str]],
    *,
    log_auto_select_warning: bool = True,
) -> List[str]:
    """
    Columns used for model features (target and sensitive excluded unless explicitly listed).
    """
    if features is not None:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(
                f"Configured feature column(s) not found in dataframe: {missing}. "
                f"Available columns: {list(df.columns)}."
            )
        return list(features)

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    exclude = {target_col, *sensitive}
    resolved = [c for c in numeric_cols if c not in exclude]
    if not resolved:
        raise ValueError(
            "No numeric feature columns remain after excluding target and sensitive attributes. "
            f"Dataframe columns: {list(df.columns)}. "
            "Set an explicit `features` list in config or add numeric predictors."
        )
    if log_auto_select_warning:
        logger.warning(
            "No explicit `features` in config; auto-selecting numeric columns for modeling "
            "(excluding target and sensitive): %s",
            resolved,
        )
    return resolved


def _strip_sensitive_columns(df: pd.DataFrame, sensitive: Sequence[str]) -> pd.DataFrame:
    drop = [c for c in sensitive if c in df.columns]
    if not drop:
        return df
    return df.drop(columns=drop)


@dataclass
class ValidationResult:
    """Result of final validation step."""

    passed: bool
    baseline_metric_value: float
    final_metric_value: float
    threshold: Optional[float]
    improvement: float  # negative means improvement (reduction in unfairness)
    message: str


@dataclass
class WorkflowResult:
    """Complete workflow execution result."""

    baseline_metrics: Dict[str, Any]
    final_metrics: Dict[str, Any]
    validation_result: ValidationResult
    model: Any
    transformed_df: pd.DataFrame
    predictions: np.ndarray
    y_test: Optional[np.ndarray] = None  # For accuracy computation
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSplit:
    """Stratified train/test partition shared across workflow steps."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    s_train: np.ndarray
    s_test: np.ndarray


def _seed_workflow_rng(random_state: int) -> None:
    """Seed numpy and PyTorch (if installed) for reproducible training."""
    np.random.seed(random_state)
    try:
        import torch

        torch.manual_seed(random_state)
    except ImportError:
        pass


def _make_workflow_split(
    df: pd.DataFrame,
    config: PipelineConfig,
    train_size: float,
    random_state: int,
    *,
    log_feature_auto_select: bool = False,
) -> WorkflowSplit:
    """Create a single stratified train/test split for the integrated workflow."""
    if config.training is None:
        raise TrainingError(
            "Configuration must include a 'training' section to build a workflow split.",
            suggestion="Add a 'training' section with 'target_column' to your configuration.",
        )

    target_col = config.training.target_column
    if target_col not in df.columns:
        raise DataValidationError(
            f"Target column '{target_col}' not found in dataframe.",
            missing_columns=[target_col],
            data_shape=df.shape,
        )

    y = df[target_col].to_numpy()
    feature_cols = _resolve_feature_columns(
        df,
        target_col,
        tuple(config.sensitive),
        config.features,
        log_auto_select_warning=log_feature_auto_select,
    )
    X = df[feature_cols]
    if len(config.sensitive) == 1:
        sensitive_features = df[config.sensitive[0]].to_numpy()
    else:
        sensitive_features = df[config.sensitive[0]].to_numpy()

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive_features,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )
    return WorkflowSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        s_train=s_train,
        s_test=s_test,
    )


def run_baseline_measurement(
    df: pd.DataFrame,
    config: PipelineConfig,
    min_group_size: int = 30,
    train_size: float = 0.8,
    random_state: int = 42,
    split: Optional[WorkflowSplit] = None,
    class_weight: str | dict | None = "balanced",
    decision_threshold: float | None = None,
    *,
    log_feature_auto_select: bool = True,
) -> Dict[str, Any]:
    """
    Execute Step 1: Baseline Measurement.

    Run fairness audit on raw input data before any transformations or training.
    When config.training and config.fairness_metric are set, trains an unconstrained
    baseline model and computes baseline fairness metrics on the test set.

    Args:
        df: Raw input dataframe
        config: Pipeline configuration
        min_group_size: Minimum group size for fairness analysis
        train_size: Proportion of data for training (used for train/test split when computing baseline metrics)
        random_state: Random seed for train/test split and baseline model
        split: Optional precomputed split (shared with ``execute_workflow``)
        class_weight: Passed to LogisticRegression. Use "balanced" for imbalanced
            datasets to prevent the model from predicting the majority class for all
            samples. Default: "balanced".
        decision_threshold: Probability threshold for converting predicted scores to
            binary predictions. If None, uses model.predict() which defaults to 0.5.
            Set to a value between 0 and 1 to simulate selective classifiers such as
            hiring screeners. Default: None (0.5 threshold).

    Returns:
        Dictionary with sensitive_attribute, data_shape, min_group_size, and optionally
        baseline_metrics (metric name -> MetricResult when training/fairness_metric are set),
        scaler (fitted StandardScaler), and decision_threshold.
    """
    log_workflow_step(logger, "baseline_measurement", status="started", n_samples=len(df))

    with PerformanceLogger(logger, "baseline_measurement", n_samples=len(df)):
        results: Dict[str, Any] = {}
        results["sensitive_attribute"] = config.sensitive
        results["data_shape"] = df.shape
        results["min_group_size"] = min_group_size

        if config.training is None or config.fairness_metric is None:
            log_workflow_step(logger, "baseline_measurement", status="completed", n_samples=len(df))
            return results

        training_cfg = config.training
        target_col = training_cfg.target_column
        if target_col not in df.columns:
            log_workflow_step(logger, "baseline_measurement", status="completed", n_samples=len(df))
            return results

        if split is None:
            split = _make_workflow_split(
                df,
                config,
                train_size,
                random_state,
                log_feature_auto_select=log_feature_auto_select,
            )

        X_train = split.X_train
        X_test = split.X_test
        y_train = split.y_train
        y_test = split.y_test
        s_test = split.s_test

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)

        baseline_model = LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight=class_weight
        )
        baseline_model.fit(X_train_scaled, y_train)
        if decision_threshold is not None:
            y_prob = baseline_model.predict_proba(X_test_scaled)[:, 1]
            baseline_predictions = (y_prob >= decision_threshold).astype(int)
        else:
            baseline_predictions = baseline_model.predict(X_test_scaled)

        results["scaler"] = scaler
        results["decision_threshold"] = decision_threshold

        analyzer = FairnessAnalyzer(min_group_size=min_group_size, backend="native")
        metric_name = config.fairness_metric
        baseline_metrics: Dict[str, Any] = {}

        if metric_name == "demographic_parity_difference":
            baseline_metrics[metric_name] = analyzer.demographic_parity_difference(
                y_pred=baseline_predictions, sensitive=s_test
            )
        elif metric_name == "equalized_odds_difference":
            test_indices = X_test.index
            if len(config.sensitive) == 1:
                baseline_metrics[metric_name] = analyzer.equalized_odds_difference(
                    y_true=y_test,
                    y_pred=baseline_predictions,
                    sensitive=s_test,
                )
            else:
                attrs_df_test = df.loc[test_indices, config.sensitive].copy()
                sensitive_test = attrs_df_test.astype(str).agg("_".join, axis=1).to_numpy().ravel()
                baseline_metrics[metric_name] = analyzer.equalized_odds_difference(
                    y_true=y_test,
                    y_pred=baseline_predictions,
                    sensitive=sensitive_test,
                    attrs_df=attrs_df_test,
                    columns=config.sensitive,
                )
        else:
            if hasattr(analyzer, metric_name):
                metric_func = getattr(analyzer, metric_name)
                baseline_metrics[metric_name] = metric_func(
                    y_true=y_test, y_pred=baseline_predictions, sensitive=s_test
                )

        results["baseline_metrics"] = baseline_metrics
        log_workflow_step(logger, "baseline_measurement", status="completed", n_samples=len(df))
        return results


def run_transform_and_train(
    df: pd.DataFrame,
    config: PipelineConfig,
    train_size: float = 0.8,
    random_state: int = 42,
    split: Optional[WorkflowSplit] = None,
    class_weight: str | dict | None = "balanced",
    decision_threshold: float | None = None,
    scaler: Optional[StandardScaler] = None,
    *,
    log_feature_auto_select: bool = True,
) -> Tuple[Any, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute Step 2: Transform Data + Train Model.

    Apply pipeline transformers, then train model using configured training method.

    Args:
        df: Input dataframe
        config: Pipeline configuration with training section
        train_size: Proportion of data for training
        random_state: Random seed for train/test split and model training
        split: Optional precomputed split (shared with ``execute_workflow``)
        class_weight: Passed to LogisticRegression base estimator for reductions.
            Default: "balanced".
        decision_threshold: Probability threshold for binary predictions. If None,
            uses model.predict() (0.5). Default: None.
        scaler: Optional baseline-fitted StandardScaler; when provided, transformed
            features are scaled with this scaler (not refit).

    Returns:
        Tuple of (trained_model, transformed_df, y_train, y_test, predictions)
    """
    _seed_workflow_rng(random_state)

    if config.training is None:
        raise TrainingError(
            "Configuration must include a 'training' section for Step 2 (Transform and Train).",
            suggestion="Add a 'training' section to your configuration file with 'method' and 'target_column' fields.",
        )

    training_cfg = config.training
    target_col = training_cfg.target_column

    if target_col not in df.columns:
        raise DataValidationError(
            f"Target column '{target_col}' not found in dataframe.",
            missing_columns=[target_col],
            data_shape=df.shape,
            suggestion=f"Ensure the dataframe contains a column named '{target_col}' or update the 'target_column' in your configuration.",
        )

    if split is None:
        split = _make_workflow_split(
            df,
            config,
            train_size,
            random_state,
            log_feature_auto_select=log_feature_auto_select,
        )

    X_train = split.X_train
    X_test = split.X_test
    y_train = split.y_train
    y_test = split.y_test
    s_train = split.s_train

    train_sample_weight: Optional[np.ndarray] = None

    # Apply pipeline transformations (sensitive columns must be present for InstanceReweighting)
    if config.pipeline:
        pipe = build_pipeline(config)
        X_train_pipe = X_train.copy()
        X_test_pipe = X_test.copy()
        for col in config.sensitive:
            if col not in df.columns:
                raise DataValidationError(
                    f"Sensitive column '{col}' not found in dataframe.",
                    missing_columns=[col],
                    data_shape=df.shape,
                )
            X_train_pipe[col] = df.loc[X_train.index, col]
            X_test_pipe[col] = df.loc[X_test.index, col]

        train_pr = apply_pipeline(pipe, X_train_pipe)
        test_pr = apply_pipeline(pipe, X_test_pipe)
        X_train_transformed = train_pr.data
        X_test_transformed = test_pr.data
        meta_train = train_pr.metadata
        if train_pr.sample_weight is not None:
            train_sample_weight = np.asarray(train_pr.sample_weight, dtype=float)
        elif meta_train and meta_train.get("sample_weight") is not None:
            train_sample_weight = np.asarray(meta_train["sample_weight"], dtype=float)
        X_train_transformed = _strip_sensitive_columns(X_train_transformed, config.sensitive)
        X_test_transformed = _strip_sensitive_columns(X_test_transformed, config.sensitive)
    else:
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()

    if train_sample_weight is not None and train_sample_weight.shape[0] != len(X_train_transformed):
        raise ValueError(
            f"sample_weight length {train_sample_weight.shape[0]} does not match "
            f"transformed training rows {len(X_train_transformed)}."
        )

    if scaler is not None:
        X_train_arr = scaler.transform(X_train_transformed.values)
        X_test_arr = scaler.transform(X_test_transformed.values)
    else:
        X_train_arr = X_train_transformed.values
        X_test_arr = X_test_transformed.values

    prob_threshold = decision_threshold if decision_threshold is not None else 0.5

    # Train model based on method
    method = training_cfg.method
    params = training_cfg.params or {}

    if method == "reductions":
        try:
            from fairlearn.reductions import DemographicParity, EqualizedOdds
        except ImportError as e:
            raise DependencyError(
                "fairlearn is required for 'reductions' training method.",
                dependency_name="fairlearn",
                extra_name="adapters",
            ) from e

        try:
            from fairness_pipeline_dev_toolkit.training import ReductionsWrapper
        except ImportError as e:
            raise DependencyError(
                "Training module is required for 'reductions' method.",
                dependency_name="training",
                extra_name="training",
            ) from e

        constraint_name = params.get("constraint", "demographic_parity")
        eps = params.get("eps", 0.01)
        T = params.get("T", 50)
        base_estimator = params.get(
            "base_estimator",
            LogisticRegression(max_iter=1000, random_state=random_state, class_weight=class_weight),
        )
        if train_sample_weight is not None:
            base_estimator = _MitigationSampleWeightMixer(base_estimator, train_sample_weight)

        if constraint_name == "demographic_parity":
            constraint = DemographicParity(difference_bound=eps)
        elif constraint_name == "equalized_odds":
            constraint = EqualizedOdds(difference_bound=eps)
        else:
            raise TrainingError(
                f"Unknown constraint: '{constraint_name}'",
                method="reductions",
                training_params={"constraint": constraint_name},
                suggestion="Use 'demographic_parity' or 'equalized_odds' as the constraint name.",
            )

        model = ReductionsWrapper(
            base_estimator=base_estimator, constraint=constraint, eps=eps, T=T
        )
        model.fit(X_train_arr, y_train, sensitive_features=s_train)

    elif method == "regularized":
        if train_sample_weight is not None:
            logger.warning(
                "Training method 'regularized' does not consume mitigation sample weights; "
                "ignoring instance reweighting weights for this run."
            )
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise DependencyError(
                "PyTorch is required for 'regularized' training method.",
                dependency_name="torch",
                extra_name="training",
            ) from e

        try:
            from fairness_pipeline_dev_toolkit.training.torch_.losses import (
                FairnessRegularizerLoss,
            )
        except ImportError as e:
            raise DependencyError(
                "Training module is required for 'regularized' method.",
                dependency_name="training",
                extra_name="training",
            ) from e

        eta = params.get("eta", 0.5)
        epochs = params.get("epochs", 10)
        lr = params.get("lr", 1e-3)
        device = params.get("device", "cpu")

        # Convert to tensors
        # Encode sensitive attributes as integers if they're strings
        if s_train.dtype == object or not np.issubdtype(s_train.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            s_train_encoded = le.fit_transform(s_train)
        else:
            s_train_encoded = s_train

        X_train_tensor = torch.tensor(X_train_arr.astype(np.float32), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.astype(np.int64), dtype=torch.int64)
        s_train_tensor = torch.tensor(s_train_encoded.astype(np.int64), dtype=torch.int64)

        # Simple neural network
        n_features = X_train_tensor.shape[1]
        model_nn = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 1))

        # Use FairnessRegularizerLoss
        criterion = FairnessRegularizerLoss(eta=eta)
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr)

        model_nn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model_nn(X_train_tensor).squeeze()
            loss = criterion(logits, y_train_tensor, s_train_tensor)
            loss.backward()
            optimizer.step()

        model = model_nn
        model.eval()

    elif method == "lagrangian":
        if train_sample_weight is not None:
            logger.warning(
                "Training method 'lagrangian' does not consume mitigation sample weights; "
                "ignoring instance reweighting weights for this run."
            )
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise DependencyError(
                "PyTorch is required for 'lagrangian' training method.",
                dependency_name="torch",
                extra_name="training",
            ) from e

        try:
            from fairness_pipeline_dev_toolkit.training import LagrangianFairnessTrainer
        except ImportError as e:
            raise DependencyError(
                "Training module is required for 'lagrangian' method.",
                dependency_name="training",
                extra_name="training",
            ) from e

        fairness_type = params.get("fairness", "demographic_parity")
        dp_tol = params.get("dp_tol", 0.02)
        eo_tol = params.get("eo_tol", 0.02)
        model_lr = params.get("model_lr", 1e-3)
        lambda_lr = params.get("lambda_lr", 1e-2)
        epochs = params.get("epochs", 10)
        batch_size = params.get("batch_size", 128)
        device = params.get("device", "cpu")

        # Convert to tensors
        # Encode sensitive attributes as integers if they're strings
        if s_train.dtype == object or not np.issubdtype(s_train.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            s_train_encoded = le.fit_transform(s_train)
        else:
            s_train_encoded = s_train

        X_train_tensor = torch.tensor(X_train_arr.astype(np.float32), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.astype(np.int64), dtype=torch.int64)
        s_train_tensor = torch.tensor(s_train_encoded.astype(np.int64), dtype=torch.int64)

        # Simple neural network
        n_features = X_train_tensor.shape[1]
        model_nn = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 1))

        trainer = LagrangianFairnessTrainer(
            model=model_nn,
            fairness=fairness_type,
            dp_tolerance=dp_tol,
            eo_tolerance=eo_tol,
            model_lr=model_lr,
            lambda_lr=lambda_lr,
            device=device,
        )
        trainer.fit(
            X_train_tensor, y_train_tensor, s_train_tensor, epochs=epochs, batch_size=batch_size
        )
        model = model_nn
        model.eval()

    else:
        raise TrainingError(
            f"Unknown training method: '{method}'",
            method=method,
            suggestion="Use one of: 'reductions', 'regularized', or 'lagrangian' as the training method.",
        )

    # Generate predictions
    if method == "reductions":
        if decision_threshold is not None:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_arr)[:, 1]
                predictions = (y_prob >= decision_threshold).astype(int)
            else:
                logger.warning(
                    "decision_threshold requires predict_proba() which is not "
                    "available on this model. Falling back to predict()."
                )
                predictions = model.predict(X_test_arr)
        else:
            predictions = model.predict(X_test_arr)
    else:
        # PyTorch models
        import torch

        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_arr.astype(np.float32), dtype=torch.float32)
            logits = model(X_test_tensor).squeeze()
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            predictions = (torch.sigmoid(logits) > prob_threshold).cpu().numpy().astype(int)

    # Combine train and test for full transformed dataframe
    transformed_df = pd.concat([X_train_transformed, X_test_transformed], ignore_index=True)
    y_full = np.concatenate([y_train, y_test])

    return model, transformed_df, y_full, y_test, predictions


def run_final_validation(
    baseline_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    config: PipelineConfig,
) -> ValidationResult:
    """
    Execute Step 3: Final Validation.

    Compare post-training fairness metrics to baseline, check against validation threshold.

    Args:
        baseline_metrics: Metrics from Step 1
        final_metrics: Metrics from Step 2 (on trained model predictions)
        config: Pipeline configuration with fairness_metric and validation_threshold

    Returns:
        ValidationResult indicating if validation passed
    """
    if config.fairness_metric is None:
        # No validation threshold specified, just report comparison
        return ValidationResult(
            passed=True,
            baseline_metric_value=0.0,
            final_metric_value=0.0,
            threshold=None,
            improvement=0.0,
            message="No fairness_metric specified in config; validation skipped.",
        )

    metric_name = config.fairness_metric

    # Extract metric values - handle both Result objects and dicts
    baseline_metric_obj = baseline_metrics.get(metric_name)
    if baseline_metric_obj is None:
        baseline_value = 0.0
    elif hasattr(baseline_metric_obj, "value"):
        baseline_value = baseline_metric_obj.value
    elif isinstance(baseline_metric_obj, dict):
        baseline_value = baseline_metric_obj.get("value", 0.0)
    else:
        baseline_value = (
            float(baseline_metric_obj) if isinstance(baseline_metric_obj, (int, float)) else 0.0
        )

    final_metric_obj = final_metrics.get(metric_name)
    if final_metric_obj is None:
        final_value = 0.0
    elif hasattr(final_metric_obj, "value"):
        final_value = final_metric_obj.value
    elif isinstance(final_metric_obj, dict):
        final_value = final_metric_obj.get("value", 0.0)
    else:
        final_value = float(final_metric_obj) if isinstance(final_metric_obj, (int, float)) else 0.0

    # Calculate improvement (negative means reduction in unfairness)
    improvement = baseline_value - final_value

    # Check threshold if specified
    threshold = config.validation_threshold
    if threshold is not None:
        passed = abs(final_value) <= threshold
        message = (
            f"Validation {'PASSED' if passed else 'FAILED'}: "
            f"{metric_name} = {final_value:.4f} "
            f"{'<=' if passed else '>'} threshold {threshold:.4f}. "
            f"Baseline: {baseline_value:.4f}, Improvement: {improvement:.4f}"
        )
    else:
        passed = True
        message = (
            f"Validation completed (no threshold): "
            f"{metric_name} = {final_value:.4f} "
            f"(baseline: {baseline_value:.4f}, improvement: {improvement:.4f})"
        )

    return ValidationResult(
        passed=passed,
        baseline_metric_value=float(baseline_value),
        final_metric_value=float(final_value),
        threshold=threshold,
        improvement=float(improvement),
        message=message,
    )


def execute_workflow(
    config: PipelineConfig,
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    min_group_size: int = 30,
    train_size: float = 0.8,
    random_state: int = 42,
    class_weight: str | dict | None = "balanced",
    decision_threshold: float | None = None,
) -> WorkflowResult:
    """
    Main orchestrator executing the complete three-step workflow.

    Args:
        config: Pipeline configuration with training section
        df: Input dataframe
        output_dir: Optional directory to save artifacts
        min_group_size: Minimum group size for fairness analysis
        train_size: Proportion of data for training
        random_state: Random seed for the stratified train/test split and model training
        class_weight: Passed to LogisticRegression. Use "balanced" for imbalanced
            datasets to prevent the model from predicting the majority class for all
            samples. Default: "balanced".
        decision_threshold: Probability threshold for converting predicted scores to
            binary predictions. If None, uses model.predict() which defaults to 0.5.
            Set to a value between 0 and 1 to simulate selective classifiers such as
            hiring screeners. Default: None (0.5 threshold).

    Returns:
        WorkflowResult with all metrics, model, and validation result
    """
    import uuid

    workflow_id = str(uuid.uuid4())[:8]

    logger.info(
        "Starting workflow execution",
        extra={"workflow_id": workflow_id, "n_samples": len(df), "random_state": random_state},
    )

    _seed_workflow_rng(random_state)
    split = _make_workflow_split(
        df, config, train_size, random_state, log_feature_auto_select=False
    )

    analyzer = FairnessAnalyzer(min_group_size=min_group_size, backend="native")

    # Step 1: Baseline Measurement
    log_workflow_step(logger, "baseline_measurement", workflow_id=workflow_id, status="started")
    logger.info("Step 1: Running baseline measurement...")
    baseline_result = run_baseline_measurement(
        df,
        config,
        min_group_size,
        train_size=train_size,
        random_state=random_state,
        split=split,
        class_weight=class_weight,
        decision_threshold=decision_threshold,
        log_feature_auto_select=True,
    )
    baseline_metrics = baseline_result.get("baseline_metrics", {})
    log_workflow_step(logger, "baseline_measurement", workflow_id=workflow_id, status="completed")

    # For baseline, we need actual predictions. If we have a target column,
    # we can compute baseline metrics on the raw data distribution
    # For now, we'll compute them after we have the model predictions to compare

    # Step 2: Transform and Train
    log_workflow_step(logger, "transform_and_train", workflow_id=workflow_id, status="started")
    logger.info("Step 2: Transforming data and training model...")
    with PerformanceLogger(
        logger, "transform_and_train", workflow_id=workflow_id, n_samples=len(df)
    ):
        model, transformed_df, y_full, y_test, predictions = run_transform_and_train(
            df,
            config,
            train_size=train_size,
            random_state=random_state,
            split=split,
            class_weight=class_weight,
            decision_threshold=decision_threshold,
            scaler=baseline_result.get("scaler"),
            log_feature_auto_select=False,
        )
    log_workflow_step(logger, "transform_and_train", workflow_id=workflow_id, status="completed")

    # Now compute baseline metrics on original data if we have target
    # For baseline, we need predictions - use a simple baseline model
    # or measure data-level fairness
    # For now, we'll use the original target distribution as a proxy
    # In practice, you might train a simple baseline model

    # Compute final metrics on test set predictions
    log_workflow_step(logger, "final_validation", workflow_id=workflow_id, status="started")
    logger.info("Step 3: Computing final metrics and validation...")

    # Sensitive attributes for final metrics (same test indices as split)
    test_indices = split.X_test.index
    if len(config.sensitive) == 1:
        sensitive_test = df.loc[test_indices, config.sensitive[0]].to_numpy()
        attrs_df_test = None
        columns = None
    else:
        attrs_df_test = df.loc[test_indices, config.sensitive].copy()
        sensitive_test = attrs_df_test.astype(str).agg("_".join, axis=1).to_numpy().ravel()
        columns = config.sensitive

    # Compute final fairness metrics
    final_metrics: Dict[str, Any] = {}
    if config.fairness_metric:
        metric_name = config.fairness_metric
        import time

        start_time = time.time()

        if metric_name == "demographic_parity_difference":
            result = analyzer.demographic_parity_difference(
                y_pred=predictions, sensitive=sensitive_test
            )
            final_metrics[metric_name] = result
            log_metric_computation(
                logger,
                metric_name=metric_name,
                value=result.value,
                n_samples=len(predictions),
                duration=time.time() - start_time,
                workflow_id=workflow_id,
            )
        elif metric_name == "equalized_odds_difference":
            result = analyzer.equalized_odds_difference(
                y_true=y_test,
                y_pred=predictions,
                sensitive=sensitive_test,
                attrs_df=attrs_df_test,
                columns=columns,
            )
            final_metrics[metric_name] = result
            log_metric_computation(
                logger,
                metric_name=metric_name,
                value=result.value,
                n_samples=len(predictions),
                duration=time.time() - start_time,
                workflow_id=workflow_id,
            )
        else:
            # Try to compute the metric if it's available
            if hasattr(analyzer, metric_name):
                metric_func = getattr(analyzer, metric_name)
                final_metrics[metric_name] = metric_func(
                    y_true=y_test, y_pred=predictions, sensitive=sensitive_test
                )

    # Step 3: Final Validation
    validation_result = run_final_validation(baseline_metrics, final_metrics, config)
    log_workflow_step(
        logger,
        "final_validation",
        workflow_id=workflow_id,
        status="completed",
        validation_passed=validation_result.passed,
        improvement=validation_result.improvement,
    )

    # Prepare artifacts
    artifacts: Dict[str, Any] = {
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "validation_result": {
            "passed": validation_result.passed,
            "baseline_metric_value": validation_result.baseline_metric_value,
            "final_metric_value": validation_result.final_metric_value,
            "threshold": validation_result.threshold,
            "improvement": validation_result.improvement,
            "message": validation_result.message,
        },
    }

    # Save artifacts if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        with open(output_path / "workflow_results.json", "w") as f:
            json.dump(artifacts, f, indent=2, default=str)

        # Save transformed data
        transformed_df.to_csv(output_path / "transformed_data.csv", index=False)

        # Save model (if possible)
        try:
            import joblib

            if hasattr(model, "predict") and not hasattr(model, "state_dict"):
                # sklearn-compatible or reductions wrapper
                joblib.dump(model, output_path / "model.pkl")
            elif hasattr(model, "state_dict"):
                # PyTorch model
                try:
                    import torch

                    torch.save(model.state_dict(), output_path / "model.pt")
                except ImportError:
                    print("Warning: PyTorch not available, cannot save model state.")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

    logger.info(
        "Workflow execution completed",
        extra={
            "workflow_id": workflow_id,
            "validation_passed": validation_result.passed,
            "improvement": validation_result.improvement,
        },
    )

    return WorkflowResult(
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        validation_result=validation_result,
        model=model,
        transformed_df=transformed_df,
        predictions=predictions,
        y_test=y_test,
        artifacts=artifacts,
    )
