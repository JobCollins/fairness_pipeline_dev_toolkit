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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


def run_baseline_measurement(
    df: pd.DataFrame, config: PipelineConfig, min_group_size: int = 30
) -> Dict[str, Any]:
    """
    Execute Step 1: Baseline Measurement.

    Run fairness audit on raw input data before any transformations or training.

    Args:
        df: Raw input dataframe
        config: Pipeline configuration
        min_group_size: Minimum group size for fairness analysis

    Returns:
        Dictionary of baseline fairness metrics
    """
    log_workflow_step(logger, "baseline_measurement", status="started", n_samples=len(df))

    with PerformanceLogger(logger, "baseline_measurement", n_samples=len(df)):
        # Check if we have predictions or need to compute baseline from data distribution
        # For baseline, we typically measure representation and statistical disparities
        # If target column exists, we can compute some metrics
        results: Dict[str, Any] = {}

        # Try to compute demographic parity if we have a target column
        # For baseline, we might not have predictions yet, so we measure data-level fairness
        # Store the sensitive attribute info for later use
        results["sensitive_attribute"] = config.sensitive
        results["data_shape"] = df.shape
        results["min_group_size"] = min_group_size

        # If target exists, we can compute some baseline metrics
        # This will be expanded when we have actual predictions in Step 3

        log_workflow_step(logger, "baseline_measurement", status="completed", n_samples=len(df))
        return results


def run_transform_and_train(
    df: pd.DataFrame,
    config: PipelineConfig,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[Any, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute Step 2: Transform Data + Train Model.

    Apply pipeline transformers, then train model using configured training method.

    Args:
        df: Input dataframe
        config: Pipeline configuration with training section
        train_size: Proportion of data for training
        random_state: Random seed for train/test split

    Returns:
        Tuple of (trained_model, transformed_df, y_train, y_test, predictions)
    """
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

    # Extract target and sensitive features
    y = df[target_col].to_numpy()
    if len(config.sensitive) == 1:
        sensitive_features = df[config.sensitive[0]].to_numpy()
    else:
        # For multiple sensitive attributes, use first one for training
        # (could be extended to support intersectional training)
        sensitive_features = df[config.sensitive[0]].to_numpy()

    # Split data - exclude target and sensitive attributes from features
    feature_cols = [col for col in df.columns if col != target_col and col not in config.sensitive]
    X = df[feature_cols]
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_features, train_size=train_size, random_state=random_state, stratify=y
    )

    # Apply pipeline transformations
    if config.pipeline:
        pipe = build_pipeline(config)
        X_train_transformed, _ = apply_pipeline(pipe, X_train)
        X_test_transformed, _ = apply_pipeline(pipe, X_test)
    else:
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()

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
        base_estimator = params.get("base_estimator", LogisticRegression(max_iter=1000))

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
        model.fit(X_train_transformed.values, y_train, sensitive_features=s_train)

    elif method == "regularized":
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

        X_train_tensor = torch.tensor(
            X_train_transformed.values.astype(np.float32), dtype=torch.float32
        )
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

        X_train_tensor = torch.tensor(
            X_train_transformed.values.astype(np.float32), dtype=torch.float32
        )
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
        predictions = model.predict(X_test_transformed.values)
    else:
        # PyTorch models
        import torch

        with torch.no_grad():
            X_test_tensor = torch.tensor(
                X_test_transformed.values.astype(np.float32), dtype=torch.float32
            )
            logits = model(X_test_tensor).squeeze()
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

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
) -> WorkflowResult:
    """
    Main orchestrator executing the complete three-step workflow.

    Args:
        config: Pipeline configuration with training section
        df: Input dataframe
        output_dir: Optional directory to save artifacts
        min_group_size: Minimum group size for fairness analysis
        train_size: Proportion of data for training

    Returns:
        WorkflowResult with all metrics, model, and validation result
    """
    import uuid

    workflow_id = str(uuid.uuid4())[:8]

    logger.info(
        "Starting workflow execution", extra={"workflow_id": workflow_id, "n_samples": len(df)}
    )

    analyzer = FairnessAnalyzer(min_group_size=min_group_size, backend="native")

    # Step 1: Baseline Measurement
    log_workflow_step(logger, "baseline_measurement", workflow_id=workflow_id, status="started")
    logger.info("Step 1: Running baseline measurement...")
    run_baseline_measurement(df, config, min_group_size)
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
            df, config, train_size=train_size
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

    # Extract sensitive attributes for final metrics from original dataframe
    # We need to reconstruct the test set indices to get the correct sensitive attributes
    from sklearn.model_selection import train_test_split

    target_col = config.training.target_column
    y = df[target_col].to_numpy()
    feature_cols = [col for col in df.columns if col != target_col and col not in config.sensitive]
    X = df[feature_cols]

    # Recreate the same split to get test indices
    _, X_test_split, _, _, _, s_test = train_test_split(
        X,
        y,
        (
            df[config.sensitive[0]].to_numpy()
            if len(config.sensitive) == 1
            else df[config.sensitive[0]].to_numpy()
        ),
        train_size=train_size,
        random_state=42,
        stratify=y,
    )

    # Get sensitive attributes for test set from original dataframe
    test_indices = X_test_split.index
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

    # Compute baseline metrics on original data (using a simple baseline model)
    # For a proper baseline, we'd train a simple model without fairness constraints
    # For now, we'll use the original target distribution
    baseline_metrics: Dict[str, Any] = {}
    if config.fairness_metric and config.training:
        # Train a simple baseline model without fairness constraints
        from sklearn.linear_model import LogisticRegression

        # Exclude target and sensitive attributes from features
        target_col = config.training.target_column
        feature_cols = [
            col for col in df.columns if col != target_col and col not in config.sensitive
        ]
        X_baseline = df[feature_cols]
        y_baseline = df[target_col].to_numpy()
        if len(config.sensitive) == 1:
            s_baseline = df[config.sensitive[0]].to_numpy()
        else:
            s_baseline = df[config.sensitive[0]].to_numpy()

        X_train_base, X_test_base, y_train_base, y_test_base, s_train_base, s_test_base = (
            train_test_split(
                X_baseline,
                y_baseline,
                s_baseline,
                train_size=train_size,
                random_state=42,
                stratify=y_baseline,
            )
        )

        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(X_train_base.values, y_train_base)
        baseline_predictions = baseline_model.predict(X_test_base.values)

        metric_name = config.fairness_metric
        if metric_name == "demographic_parity_difference":
            baseline_metrics[metric_name] = analyzer.demographic_parity_difference(
                y_pred=baseline_predictions, sensitive=s_test_base
            )
        elif metric_name == "equalized_odds_difference":
            baseline_metrics[metric_name] = analyzer.equalized_odds_difference(
                y_true=y_test_base,
                y_pred=baseline_predictions,
                sensitive=s_test_base,
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
