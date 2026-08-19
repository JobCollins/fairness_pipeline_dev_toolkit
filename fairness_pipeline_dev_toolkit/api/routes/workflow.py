from __future__ import annotations

import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from fairness_pipeline_dev_toolkit.integration.orchestrator import execute_workflow
from fairness_pipeline_dev_toolkit.io import load_data
from fairness_pipeline_dev_toolkit.pipeline.config import load_config

from ..models.responses import WorkflowResponse
from ..store import ResultStore
from .validate import _safe_float, get_store

router = APIRouter()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dict of Result dataclasses (or plain values) to JSON-safe dicts."""
    out: Dict[str, Any] = {}
    for key, val in metrics.items():
        if hasattr(val, "value") and hasattr(val, "metric"):
            # metrics.core.Result dataclass
            ci: Optional[list] = None
            if val.ci is not None:
                ci = [_safe_float(val.ci[0]), _safe_float(val.ci[1])]
            n_per_group: Optional[dict] = None
            if val.n_per_group is not None:
                n_per_group = {str(k): int(v) for k, v in val.n_per_group.items()}
            out[key] = {
                "metric": val.metric,
                "value": _safe_float(val.value),
                "ci": ci,
                "effect_size": _safe_float(val.effect_size),
                "n_per_group": n_per_group,
                # Same MetricResult.caveat field as LLM eval provenance (null for classifier metrics).
                "caveat": getattr(val, "caveat", None),
            }
        else:
            try:
                f = float(val)
                out[key] = None if (math.isnan(f) or math.isinf(f)) else f
            except (TypeError, ValueError):
                out[key] = str(val)
    return out


@router.post("/workflow", response_model=WorkflowResponse)
async def workflow_run(
    file: UploadFile = File(...),
    config: str = Form(...),
    min_group_size: int = Form(30),
    train_size: float = Form(0.8),
    random_state: int = Form(42),
    store: ResultStore = Depends(get_store),
):
    run_id = str(uuid.uuid4())
    ts = _utc_now()

    contents = await file.read()
    filename = file.filename or "upload.csv"
    suffix = ".parquet" if filename.endswith((".parquet", ".pq")) else ".csv"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        df = load_data(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    cfg = load_config(text=config)

    wf_result = execute_workflow(
        config=cfg,
        df=df,
        output_dir=None,
        min_group_size=min_group_size,
        train_size=train_size,
        random_state=random_state,
    )

    vr = wf_result.validation_result
    validation_dict: Dict[str, Any] = {
        "passed": vr.passed,
        "message": vr.message,
        "improvement": _safe_float(vr.improvement),
        "baseline_metric_value": _safe_float(vr.baseline_metric_value),
        "final_metric_value": _safe_float(vr.final_metric_value),
        "threshold": _safe_float(vr.threshold),
    }

    baseline = _serialize_metrics(wf_result.baseline_metrics)
    final = _serialize_metrics(wf_result.final_metrics)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "status": "success",
        "validation": validation_dict,
        "baseline_metrics": baseline,
        "final_metrics": final,
        "timestamp": ts,
        "_endpoint": "/workflow",
    }
    store.put(run_id, result)

    return WorkflowResponse(
        run_id=run_id,
        validation=validation_dict,
        baseline_metrics=baseline,
        final_metrics=final,
        timestamp=ts,
    )
