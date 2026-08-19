"""Shared fixtures for LLM eval tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.client import (
    AnthropicClient,
    OpenAICompatibleClient,
)


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
