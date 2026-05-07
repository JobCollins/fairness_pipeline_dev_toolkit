from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ValidateResponse(BaseModel):
    run_id: str
    status: str = "success"
    passed: bool
    metrics: Dict[str, Any]
    timestamp: str


class PipelineResponse(BaseModel):
    run_id: str
    status: str = "success"
    detector_report: Dict[str, Any]
    transformed_rows: int
    transformers_applied: List[str]
    timestamp: str


class WorkflowResponse(BaseModel):
    run_id: str
    status: str = "success"
    validation: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    final_metrics: Dict[str, Any]
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    run_id: Optional[str] = None


class ResultResponse(BaseModel):
    run_id: str
    endpoint: str
    result: Dict[str, Any]
    created_at: str
