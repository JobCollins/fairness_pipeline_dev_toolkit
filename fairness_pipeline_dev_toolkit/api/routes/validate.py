from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

from ..models.requests import ValidateRequest
from ..models.responses import ValidateResponse
from ..store import ResultStore

router = APIRouter()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v) -> Optional[float]:
    """Convert numpy scalar to Python float; return None for NaN/inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _result_to_dict(r) -> Dict[str, Any]:
    """Serialize a metrics.core.Result dataclass to a JSON-safe dict."""
    ci = None
    if r.ci is not None:
        ci = [_safe_float(r.ci[0]), _safe_float(r.ci[1])]
    n_per_group = None
    if r.n_per_group is not None:
        n_per_group = {str(k): int(v) for k, v in r.n_per_group.items()}
    return {
        "metric": r.metric,
        "value": _safe_float(r.value),
        "ci": ci,
        "effect_size": _safe_float(r.effect_size),
        "n_per_group": n_per_group,
        # Same MetricResult.caveat field as LLM eval provenance (null for classifier metrics).
        "caveat": getattr(r, "caveat", None),
    }


def get_store(request: Any = None) -> ResultStore:
    # Replaced by app.dependency_overrides in create_app()
    raise RuntimeError("ResultStore dependency not configured")  # pragma: no cover


@router.post("/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest, store: ResultStore = Depends(get_store)):
    run_id = str(uuid.uuid4())
    ts = _utc_now()

    analyzer = FairnessAnalyzer(min_group_size=req.min_group_size, backend=req.backend)
    y_pred = np.array(req.y_pred)
    sensitive = np.array(req.sensitive)

    metrics: Dict[str, Any] = {}

    dpd = analyzer.demographic_parity_difference(
        y_pred=y_pred,
        sensitive=sensitive,
        with_ci=req.with_ci,
        ci_level=req.ci_level,
        with_effect_size=req.with_effects,
    )
    metrics["demographic_parity_difference"] = _result_to_dict(dpd)

    if req.y_true is not None:
        y_true = np.array(req.y_true)
        eod = analyzer.equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            with_ci=req.with_ci,
            ci_level=req.ci_level,
            with_effect_size=req.with_effects,
        )
        metrics["equalized_odds_difference"] = _result_to_dict(eod)

    if req.y_score is not None and req.y_true is not None:
        y_score = np.array(req.y_score)
        mae = analyzer.mae_parity_difference(
            y_true=np.array(req.y_true),
            y_pred=y_score,
            sensitive=sensitive,
            with_ci=req.with_ci,
            ci_level=req.ci_level,
            with_effect_size=req.with_effects,
        )
        metrics["mae_parity_difference"] = _result_to_dict(mae)

    dpd_value = metrics["demographic_parity_difference"]["value"]
    passed = True
    if req.threshold is not None and dpd_value is not None:
        passed = abs(dpd_value) <= req.threshold

    result = {
        "run_id": run_id,
        "status": "success",
        "passed": passed,
        "metrics": metrics,
        "timestamp": ts,
        "_endpoint": "/validate",
    }
    store.put(run_id, result)

    return ValidateResponse(
        run_id=run_id,
        passed=passed,
        metrics=metrics,
        timestamp=ts,
    )


@router.get("/results/{run_id}")
async def get_result(run_id: str, store: ResultStore = Depends(get_store)):
    data = store.get(run_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "NotFound", "message": f"No result found for run_id: {run_id}"},
        )
    endpoint = data.pop("_endpoint", "/validate")
    created_at = data.get("timestamp", "")
    return {
        "run_id": run_id,
        "endpoint": endpoint,
        "result": data,
        "created_at": created_at,
    }
