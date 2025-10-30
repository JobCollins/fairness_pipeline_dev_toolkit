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

import io
import json
from dataclasses import asdict, is_dataclass
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
                    f"{prefix}.{name}.{key}", json.dumps(res_dict[key], ensure_ascii=False)
                )
        aggregate_results[name] = res_dict

    # Log a single JSON artifact with all results
    json_bytes = json.dumps(aggregate_results, indent=2, ensure_ascii=False).encode("utf-8")
    with io.BytesIO(json_bytes) as artifact_file:
        mlflow.log_artifact_local = getattr(mlflow, "log_artifact_local", None)
        # use log_text when available (mlflow >-2.8. has it)
        log_text = getattr(mlflow, "log_text", None)
        if log_text is not None:
            log_text(json_bytes.decode("utf-8"), artifact_file or f"{prefix}_results.json")
        else:
            # Fallback: write a temporary file via mlflow's log_artifact - requires a file path
            # users with older mlflow versions may skip this step; metrics/params still logged
            pass

    # Optionally log a human-readable md report as artifact
    if artifact_name and artifact_content is not None:
        log_text = getattr(mlflow, "log_text", None)
        if log_text is not None:
            log_text(artifact_content, artifact_file=artifact_name)

    return True
