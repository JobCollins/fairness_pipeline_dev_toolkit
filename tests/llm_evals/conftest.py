"""Shared fixtures for LLM eval tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
import yaml

from fairness_pipeline_dev_toolkit.llm_evals.client import (
    AnthropicClient,
    OpenAICompatibleClient,
)
from fairness_pipeline_dev_toolkit.llm_evals.config import LLMEvalConfig


def write_llm_eval_yaml(path: Path, config: LLMEvalConfig) -> Path:
    """Write an ``llm_eval:`` YAML file from an ``LLMEvalConfig`` (CLI / harness tests)."""
    block: Dict[str, Any] = {
        "provider": config.provider,
        "model": config.model,
        "evaluators": list(config.evaluators),
        "params": dict(config.params),
    }
    if config.cache_dir:
        block["cache_dir"] = config.cache_dir
    if config.counterfactual is not None:
        block["counterfactual"] = {
            "template": config.counterfactual.template,
            "dimensions": config.counterfactual.dimensions,
            "defaults": config.counterfactual.defaults,
        }
    if config.bbq_path:
        block["bbq_path"] = config.bbq_path
    if config.allow_small_samples:
        block["allow_small_samples"] = True
    if config.max_requests_per_run is not None:
        block["max_requests_per_run"] = config.max_requests_per_run
    path.write_text(yaml.safe_dump({"llm_eval": block}), encoding="utf-8")
    return path


@pytest.fixture
def mlflow_sqlite_tracking(tmp_path):
    """Local MLflow tracking URI that works after FileStore was put in maintenance mode.

    Newer MLflow raises unless ``MLFLOW_ALLOW_FILE_STORE=true``. Tests use sqlite instead
    so they stay compatible without opting into a deprecated backend.
    """
    import mlflow

    uri = "sqlite:///" + (tmp_path / "mlflow.db").resolve().as_posix()
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield uri
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def assert_no_live_llm_calls(monkeypatch):
    """Patch provider live-call sites and assert they were never invoked.

    Use on every default-path cache-replay test. Do not use on ``live_llm``
    populate tests (real provider calls) or tests that intentionally mock a local responder.
    """
    openai_spy = AsyncMock(side_effect=RuntimeError("unexpected live OpenAI SDK call"))
    anthropic_spy = AsyncMock(side_effect=RuntimeError("unexpected live Anthropic SDK call"))
    monkeypatch.setattr(OpenAICompatibleClient, "_complete_uncached", openai_spy)
    monkeypatch.setattr(AnthropicClient, "_complete_uncached", anthropic_spy)
    yield
    assert openai_spy.await_count == 0, "OpenAI _complete_uncached was invoked"
    assert anthropic_spy.await_count == 0, "Anthropic _complete_uncached was invoked"
