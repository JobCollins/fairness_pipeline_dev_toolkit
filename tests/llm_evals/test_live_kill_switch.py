"""Live-call kill-switch: live HTTP is forbidden unless opted in."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.client import (
    FAIRPIPE_LLM_ALLOW_LIVE,
    FAIRPIPE_LLM_FORBID_LIVE,
    AnthropicClient,
    LiveLLMCallForbidden,
    OpenAICompatibleClient,
    allow_live_llm_calls,
)
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
)
from fairness_pipeline_dev_toolkit.llm_evals.runner import run_llm_eval


def test_jupyter_like_unset_env_fails_instantly(monkeypatch):
    """The original incident: Jupyter, no pytest, missing cache_dir must not hang."""
    monkeypatch.delenv(FAIRPIPE_LLM_FORBID_LIVE, raising=False)
    monkeypatch.delenv(FAIRPIPE_LLM_ALLOW_LIVE, raising=False)
    client = AnthropicClient("claude-haiku-4-5", api_key="sk-not-used")
    t0 = time.perf_counter()
    with pytest.raises(LiveLLMCallForbidden, match="forbidden by default"):
        asyncio.run(client.complete("this must not hit the network"))
    assert time.perf_counter() - t0 < 2.0


def test_forbid_live_raises_instantly_at_sdk_call_site():
    openai = OpenAICompatibleClient("gpt-4o-mini", api_key="sk-not-used")
    anthropic = AnthropicClient("claude-haiku-4-5", api_key="sk-not-used")

    t0 = time.perf_counter()
    with pytest.raises(LiveLLMCallForbidden, match="FAIRPIPE_LLM_ALLOW_LIVE"):
        asyncio.run(openai.complete("this must not hit the network"))
    with pytest.raises(LiveLLMCallForbidden, match="FAIRPIPE_LLM_ALLOW_LIVE"):
        asyncio.run(anthropic.complete("this must not hit the network"))
    assert time.perf_counter() - t0 < 2.0


def test_forbid_live_missing_cache_dir_fails_instantly():
    """Regression: no cache_dir would otherwise attempt a live call and hang."""
    config = LLMEvalConfig(
        provider="anthropic",
        model="claude-haiku-4-5",
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual=CounterfactualConfig(
            template="Write a hiring recommendation for {name}, a {gender} engineer.",
            dimensions={"gender": ["woman", "man"]},
            defaults={"name": "Alex"},
        ),
        cache_dir=None,
        params={"temperature": 0.0, "max_tokens": 16},
    )
    t0 = time.perf_counter()
    with pytest.raises(LiveLLMCallForbidden, match="cache_dir"):
        run_llm_eval(config, with_ci=False, allow_small_samples=True)
    assert time.perf_counter() - t0 < 2.0


def test_allow_live_context_manager_opts_in(monkeypatch):
    monkeypatch.delenv(FAIRPIPE_LLM_ALLOW_LIVE, raising=False)
    monkeypatch.delenv(FAIRPIPE_LLM_FORBID_LIVE, raising=False)
    assert os.getenv(FAIRPIPE_LLM_ALLOW_LIVE) is None
    with allow_live_llm_calls():
        assert os.environ[FAIRPIPE_LLM_ALLOW_LIVE] == "1"
    assert os.getenv(FAIRPIPE_LLM_ALLOW_LIVE) is None
