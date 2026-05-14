from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from fairness_pipeline_dev_toolkit.io import load_data
from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
    run_detectors,
)

from ..models.responses import PipelineResponse
from ..store import ResultStore
from .validate import get_store

router = APIRouter()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/pipeline", response_model=PipelineResponse)
async def pipeline_run(
    file: UploadFile = File(...),
    config: str = Form(...),
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

    bias_report = run_detectors(df, cfg)

    try:
        pipe = build_pipeline(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    Xt = apply_pipeline(pipe, df).data
    transformers_applied = [name for name, _ in pipe.steps]

    detector_dict = bias_report.to_dict()

    result: dict[str, Any] = {
        "run_id": run_id,
        "status": "success",
        "detector_report": detector_dict,
        "transformed_rows": len(Xt),
        "transformers_applied": transformers_applied,
        "timestamp": ts,
        "_endpoint": "/pipeline",
    }
    store.put(run_id, result)

    return PipelineResponse(
        run_id=run_id,
        detector_report=detector_dict,
        transformed_rows=len(Xt),
        transformers_applied=transformers_applied,
        timestamp=ts,
    )
