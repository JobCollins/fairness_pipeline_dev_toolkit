from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .routes import health_router, pipeline_router, validate_router, workflow_router
from .routes.validate import get_store
from .store import ResultStore

logger = logging.getLogger(__name__)

_API_VERSION = "0.7.0"


def create_app() -> FastAPI:
    """Factory that creates and configures the FastAPI application."""
    app = FastAPI(
        title="fairpipe API",
        version=_API_VERSION,
        description=(
            "REST API for the fairpipe fairness toolkit. "
            "Compute fairness metrics, run bias pipelines, and execute full workflows over HTTP."
        ),
    )

    # Singleton store attached to app state
    store = ResultStore()
    app.state.store = store

    # Wire the store into the dependency system
    def _get_store() -> ResultStore:
        return app.state.store

    app.dependency_overrides[get_store] = _get_store

    # Register routers
    app.include_router(health_router)
    app.include_router(validate_router)
    app.include_router(pipeline_router)
    app.include_router(workflow_router)

    # Global exception handler — returns 500 for any unhandled exception.
    # Note: HTTPException is handled by FastAPI before reaching this handler.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError", "message": str(exc), "run_id": None},
        )

    return app
