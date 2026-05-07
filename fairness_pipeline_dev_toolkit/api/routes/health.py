from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from ..models.responses import HealthResponse

router = APIRouter()

_API_VERSION = "0.7.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=_API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
