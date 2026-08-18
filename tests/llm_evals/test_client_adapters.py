"""Tests for LLMClient provider adapters (mocked — no live network)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.cache import ResponseCache
from fairness_pipeline_dev_toolkit.llm_evals.client import (
    AnthropicClient,
    LocalLLMClient,
    OpenAICompatibleClient,
    RateLimitError,
    get_llm_client,
)


def _openai_response(text: str):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _anthropic_response(text: str):
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def test_openai_adapter_uses_mock_client():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_openai_response("hello"))

    client = OpenAICompatibleClient(
        "gpt-4o-mini",
        api_key="test-key",
        client=mock_client,
    )
    assert client.available() is True

    result = asyncio.run(client.complete("Say hi", params={"temperature": 0.0}))
    assert result == "hello"
    mock_client.chat.completions.create.assert_awaited_once()


def test_anthropic_adapter_uses_mock_client():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_anthropic_response("anthropic-hi"))

    client = AnthropicClient(
        "claude-3-haiku-20240307",
        api_key="test-key",
        client=mock_client,
    )
    assert client.available() is True

    result = asyncio.run(client.complete("Say hi"))
    assert result == "anthropic-hi"
    mock_client.messages.create.assert_awaited_once()


def test_local_client_no_credentials_required():
    client = LocalLLMClient("local-model")
    assert client.available() is True
    result = asyncio.run(client.complete("prompt"))
    assert "[local:local-model]" in result


def test_openai_available_false_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        client = OpenAICompatibleClient("gpt-4o-mini")
        assert client.available() is False


def test_anthropic_available_false_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        client = AnthropicClient("claude-3-haiku-20240307")
        assert client.available() is False


def test_openai_retries_on_rate_limit():
    mock_client = MagicMock()
    call_count = {"n": 0}

    async def _create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            exc = Exception("429 rate limit")
            exc.status_code = 429
            raise exc
        return _openai_response("after-retry")

    mock_client.chat.completions.create = _create

    client = OpenAICompatibleClient(
        "gpt-4o-mini",
        api_key="test-key",
        client=mock_client,
        retry_base_seconds=0.001,
    )

    result = asyncio.run(client.complete("retry me"))
    assert result == "after-retry"
    assert call_count["n"] == 2


def test_openai_raises_rate_limit_after_max_retries():
    mock_client = MagicMock()

    async def _always_429(**kwargs):
        raise RateLimitError("429")

    mock_client.chat.completions.create = _always_429

    client = OpenAICompatibleClient(
        "gpt-4o-mini",
        api_key="test-key",
        client=mock_client,
        max_retries=2,
        retry_base_seconds=0.001,
    )

    with pytest.raises(RateLimitError):
        asyncio.run(client.complete("fail"))


def test_complete_uses_cache(tmp_path):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_openai_response("cached"))

    cache = ResponseCache(cache_dir=tmp_path)
    client = OpenAICompatibleClient(
        "gpt-4o-mini",
        api_key="test-key",
        client=mock_client,
        cache=cache,
    )

    first = asyncio.run(client.complete("same prompt", params={"temperature": 0.0}))
    second = asyncio.run(client.complete("same prompt", params={"temperature": 0.0}))

    assert first == second == "cached"
    mock_client.chat.completions.create.assert_awaited_once()
    assert cache.hits == 1


def test_complete_batch_concurrent():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[_openai_response("a"), _openai_response("b")]
    )

    client = OpenAICompatibleClient(
        "gpt-4o-mini",
        api_key="test-key",
        client=mock_client,
        batch_concurrency=2,
    )

    results = asyncio.run(client.complete_batch(["one", "two"]))
    assert results == ["a", "b"]
    assert mock_client.chat.completions.create.await_count == 2


def test_get_llm_client_factory():
    local = get_llm_client("local", "m")
    assert isinstance(local, LocalLLMClient)

    openai = get_llm_client("openai", "gpt-4o-mini", api_key="k", client=MagicMock())
    assert isinstance(openai, OpenAICompatibleClient)

    anthropic = get_llm_client("anthropic", "claude", api_key="k", client=MagicMock())
    assert isinstance(anthropic, AnthropicClient)


def test_get_llm_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_llm_client("unknown", "model")


def test_replay_only_raises_on_cache_miss(tmp_path):
    from fairness_pipeline_dev_toolkit.llm_evals.client import CacheMissError

    cache = ResponseCache(tmp_path)
    client = LocalLLMClient("m", cache=cache, replay_only=True)
    with pytest.raises(CacheMissError, match="replay-only"):
        asyncio.run(client.complete("uncached prompt"))
