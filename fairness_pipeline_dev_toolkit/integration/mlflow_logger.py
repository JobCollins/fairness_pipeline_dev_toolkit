"""
MLflow logger utilities.
Design goals:
- Be safe to import/run even if MLflow is not installed (graceful degradation).
- Accept MetricResult objects or plain dicts.
- Log scalar metrics to MLflow metrics; structured blobs as JSON artifacts.

Typical usage:
    from fairness_pipeline_dev_toolkit.integration.mlflow_logger import log_fairness_metrics
    from fairness_pipeline_dev_toolkit.integration.reporting import to_markdown_report
    results = {
        "demographic_parity_difference": MetricResult(...),
        "equalized_odds_difference": MetricResult(...),
    }
    ok = log_fairness_metrics(results, artifact_name="fairness_report.md", artifact_content=to_markdown_report(results))
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _is_mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def _coerce_result_to_dict(val: Any) -> Dict[str, Any]:
    if is_dataclass(val):
        return asdict(val)
    if isinstance(val, Mapping):
        return dict(val)
    # Fallback: treat as a scalar metric
    return {"value": float(val) if isinstance(val, (int, float)) else val}


def log_fairness_metrics(
    results: Mapping[str, Any],
    *,
    prefix: str = "fairness_",
    artifact_name: Optional[str] = None,
    artifact_content: Optional[str] = None,
) -> bool:
    """
    Log fairness into the active MLflow run, if MLflow is available.

    Args:
        results: Mapping of metric names to MetricResult objects or dicts.
        prefix: Prefix to add to metric names in MLflow.
        artifact_name: Optional name for an artifact to log (e.g., "fairness_report.md").
        artifact_content: Content of the artifact to log, if artifact_name is provided.
    Returns:
        bool: True if MLflow was available and logging was performed. False if MLflow is not available.
    """
    if not _is_mlflow_available():
        return False

    import mlflow

    # log all scalars to metrics; structured fields to params and an aggregate JSON artifact
    aggregate_results: Dict[str, Any] = {}

    for name, val in results.items():
        res_dict = _coerce_result_to_dict(val)
        # Scalar value (if present) goes to metrics
        if (
            "value" in res_dict
            and isinstance(res_dict["value"], (int, float))
            and res_dict["value"] == res_dict["value"]
        ):
            mlflow.log_metric(f"{prefix}.{name}.value", float(res_dict["value"]))

        # confidence interval, effect sizes, counts, etc. go to params (stringified) and artifact blob
        for key in ("ci", "effect_size", "n_per_group"):
            if key in res_dict and res_dict[key] is not None:
                mlflow.log_param(
                    f"{prefix}.{name}.{key}",
                    json.dumps(res_dict[key], ensure_ascii=False),
                )
        aggregate_results[name] = res_dict

    # Log a single JSON artifact with all results
    artifact_filename = f"{prefix}results.json" if prefix else "results.json"
    log_dict = getattr(mlflow, "log_dict", None)
    if callable(log_dict):
        log_dict(aggregate_results, artifact_filename)
    else:
        payload = json.dumps(aggregate_results, indent=2, ensure_ascii=False)
        log_text = getattr(mlflow, "log_text", None)
        if callable(log_text):
            log_text(payload, artifact_filename)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                tmp.write(payload)
                tmp.flush()
                tmp_path = tmp.name
            log_artifact = getattr(mlflow, "log_artifact", None)
            try:
                if callable(log_artifact):
                    log_artifact(tmp_path, artifact_path=None)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # Optionally log a human-readable md report as artifact
    if artifact_name and artifact_content is not None:
        log_text = getattr(mlflow, "log_text", None)
        if callable(log_text):
            log_text(artifact_content, artifact_file=artifact_name)

    return True


def log_workflow_results(
    workflow_result: Any,
    *,
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> bool:
    """
    Log complete workflow results to MLflow, including accuracy, model, and config.

    Args:
        workflow_result: WorkflowResult object from execute_workflow
        config_path: Path to config.yml file to log as artifact
        experiment_name: MLflow experiment name (creates new run in this experiment)
        run_name: Optional name for the MLflow run

    Returns:
        bool: True if MLflow was available and logging was performed
    """
    if not _is_mlflow_available():
        return False

    import mlflow

    # Set experiment if specified
    if experiment_name:
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            # Experiment might not exist, create it
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

    # Start run
    with mlflow.start_run(run_name=run_name):
        # Log baseline metrics with prefix
        if workflow_result.baseline_metrics:
            log_fairness_metrics(workflow_result.baseline_metrics, prefix="baseline_")

        # Log final metrics with prefix
        if workflow_result.final_metrics:
            log_fairness_metrics(workflow_result.final_metrics, prefix="final_")

        # Log primary fairness metric (if specified in config)
        if workflow_result.final_metrics:
            # Extract primary metric value
            for metric_name, metric_value in workflow_result.final_metrics.items():
                res_dict = _coerce_result_to_dict(metric_value)
                if "value" in res_dict and isinstance(res_dict["value"], (int, float)):
                    mlflow.log_metric("primary_fairness_metric", float(res_dict["value"]))

        # Log accuracy (primary performance metric)
        # Compute accuracy from predictions if we have y_test
        if hasattr(workflow_result, "y_test") and workflow_result.y_test is not None:
            import numpy as np

            y_test = workflow_result.y_test
            predictions = workflow_result.predictions
            if len(y_test) == len(predictions):
                accuracy = float(np.mean(y_test == predictions))
                mlflow.log_metric("accuracy", accuracy)

        # Log validation result
        if workflow_result.validation_result:
            vr = workflow_result.validation_result
            mlflow.log_metric("validation_passed", 1.0 if vr.passed else 0.0)
            mlflow.log_metric("baseline_metric_value", vr.baseline_metric_value)
            mlflow.log_metric("final_metric_value", vr.final_metric_value)
            mlflow.log_metric("improvement", vr.improvement)
            if vr.threshold is not None:
                mlflow.log_metric("validation_threshold", vr.threshold)
            mlflow.log_param("validation_message", vr.message)

        # Log model artifact
        if workflow_result.model:
            try:
                import joblib

                if hasattr(workflow_result.model, "predict") and not hasattr(
                    workflow_result.model, "state_dict"
                ):
                    # sklearn-compatible or reductions wrapper
                    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                        joblib.dump(workflow_result.model, tmp.name)
                        tmp_path = tmp.name
                    try:
                        mlflow.log_artifact(tmp_path, artifact_path="model")
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                elif hasattr(workflow_result.model, "state_dict"):
                    # PyTorch model
                    try:
                        import torch

                        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                            torch.save(workflow_result.model.state_dict(), tmp.name)
                            tmp_path = tmp.name
                        try:
                            mlflow.log_artifact(tmp_path, artifact_path="model")
                        finally:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                    except ImportError:
                        pass  # PyTorch not available
            except Exception as e:
                # Log warning but continue
                print(f"Warning: Could not log model artifact: {e}")

        # Log config.yml as artifact
        if config_path and Path(config_path).exists():
            mlflow.log_artifact(config_path, artifact_path="config")

        # Log workflow results JSON
        if hasattr(workflow_result, "artifacts"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(
                    workflow_result.artifacts,
                    tmp,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
                tmp_path = tmp.name
            try:
                mlflow.log_artifact(tmp_path, artifact_path="workflow")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return True


def log_llm_eval_results(
    results: Mapping[str, Any],
    *,
    artifact_name: Optional[str] = None,
    artifact_content: Optional[str] = None,
) -> bool:
    """Log LLM eval MetricResults the same way ``log_fairness_metrics`` logs classifier results."""
    ok = log_fairness_metrics(
        results,
        prefix="llm_eval_",
        artifact_name=artifact_name,
        artifact_content=artifact_content,
    )
    if not ok:
        return False
    import mlflow

    for name, val in results.items():
        res_dict = _coerce_result_to_dict(val)
        caveat = res_dict.get("caveat")
        if caveat:
            mlflow.set_tag(f"llm_eval.{name}.caveat", str(caveat))
    return True
