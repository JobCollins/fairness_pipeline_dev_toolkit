from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from fairness_pipeline_dev_toolkit import __version__ as _pkg_version

from ..models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=_pkg_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
