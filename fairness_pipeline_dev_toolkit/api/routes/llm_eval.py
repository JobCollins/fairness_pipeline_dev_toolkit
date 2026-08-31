from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import yaml
from fastapi import APIRouter, Depends, HTTPException

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError
from fairness_pipeline_dev_toolkit.llm_evals.client import (
    CacheMissError,
    LiveLLMCallForbidden,
)
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    LLMEvalConfig,
    load_llm_eval_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.gating import evaluate_llm_eval_gate
from fairness_pipeline_dev_toolkit.llm_evals.runner import run_llm_eval_async

from ..models.requests import LLMEvalRequest
from ..models.responses import LLMEvalResponse
from ..store import ResultStore
from .validate import _result_to_dict, get_store

router = APIRouter()

_REST_ONLY_FIELDS = frozenset(
    {
        "min_group_size",
        "with_ci",
        "ci_level",
        "threshold",
        "metric",
        "bootstrap_B",
        "config",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_422(message: str, *, error: str = "ConfigValidationError") -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"error": error, "message": message, "run_id": None},
    )


def _config_from_request(req: LLMEvalRequest) -> LLMEvalConfig:
    """Build ``LLMEvalConfig`` via ``load_llm_eval_config`` (credential rejection included)."""
    provided: Dict[str, Any] = req.model_dump(exclude_unset=True)
    provided.update(dict(req.model_extra or {}))

    yaml_text = provided.pop("config", None)
    for key in _REST_ONLY_FIELDS:
        provided.pop(key, None)

    if yaml_text is not None:
        if not isinstance(yaml_text, str) or not yaml_text.strip():
            raise ConfigValidationError("Field 'config' must be a non-empty YAML string.")
        try:
            root = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"Invalid YAML in 'config': {exc}") from exc
        if not isinstance(root, dict):
            raise ConfigValidationError("YAML 'config' must parse to a mapping.")
        if "llm_eval" in root:
            block = root["llm_eval"]
            if not isinstance(block, dict):
                raise ConfigValidationError("Config field 'llm_eval' must be a mapping.")
            block = dict(block)
        else:
            block = dict(root)
        block.update(provided)
        return load_llm_eval_config(obj=block)

    return load_llm_eval_config(obj=provided)


@router.post("/llm-eval", response_model=LLMEvalResponse)
async def llm_eval(req: LLMEvalRequest, store: ResultStore = Depends(get_store)):
    """Run LLM fairness evaluators. Credentials are env-only on the server process."""
    run_id = str(uuid.uuid4())
    ts = _utc_now()

    try:
        config = _config_from_request(req)
    except (ConfigValidationError, TypeError, ValueError) as exc:
        raise _http_422(str(exc)) from exc

    if req.metric is not None and req.metric not in config.evaluators:
        raise _http_422(
            f"metric {req.metric!r} is not in this run's evaluators {config.evaluators}."
        )

    try:
        result = await run_llm_eval_async(
            config,
            min_group_size=req.min_group_size,
            with_ci=req.with_ci,
            ci_level=req.ci_level,
            bootstrap_B=req.bootstrap_B,
        )
    except CacheMissError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "CacheMiss", "message": str(exc), "run_id": None},
        ) from exc
    except LiveLLMCallForbidden as exc:
        raise HTTPException(
            status_code=403,
            detail={"error": "LiveLLMCallForbidden", "message": str(exc), "run_id": None},
        ) from exc
    except ConfigValidationError as exc:
        raise _http_422(str(exc)) from exc

    metrics: Dict[str, Any] = {
        name: _result_to_dict(metric) for name, metric in result.metrics.items()
    }

    try:
        gate_status, passed = evaluate_llm_eval_gate(
            metrics, threshold=req.threshold, metric=req.metric
        )
    except KeyError as exc:
        raise _http_422(f"metric {req.metric!r} was not present in the eval results.") from exc

    stored: Dict[str, Any] = {
        "run_id": run_id,
        "status": "success",
        "gate_status": gate_status,
        "passed": passed,
        "metrics": metrics,
        "timestamp": ts,
        "_endpoint": "/llm-eval",
    }
    store.put(run_id, stored)

    return LLMEvalResponse(
        run_id=run_id,
        gate_status=gate_status,
        passed=passed,
        metrics=metrics,
        timestamp=ts,
    )
