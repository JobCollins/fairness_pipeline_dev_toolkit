"""MLflow logging for LLM evals uses a local tracking URI (no live server)."""

from __future__ import annotations

import pytest

from fairness_pipeline_dev_toolkit.integration.mlflow_logger import log_llm_eval_results
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


def test_log_llm_eval_results_local_tracking_uri(mlflow_sqlite_tracking):
    import mlflow

    mlflow.set_experiment("llm_eval_local")
    metrics = {
        "refusal_rate_disparity": MetricResult(
            metric="refusal_rate_disparity",
            value=0.12,
            ci=(0.10, 0.14),
            effect_size=0.12,
            n_per_group={"woman": 9, "man": 9},
        )
    }
    with mlflow.start_run() as run:
        ok = log_llm_eval_results(metrics, artifact_name="llm_report.md", artifact_content="# ok")
        run_id = run.info.run_id
    assert ok is True

    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    assert data.metrics["llm_eval_.refusal_rate_disparity.value"] == pytest.approx(0.12)
