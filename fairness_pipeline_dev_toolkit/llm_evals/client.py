from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

from fairness_pipeline_dev_toolkit.exceptions import (
    DependencyError,
    FairnessToolkitError,
)

from .cache import ResponseCache, make_cache_key

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
# Live provider HTTP is forbidden unless FAIRPIPE_LLM_ALLOW_LIVE is set.
# Default-on so Jupyter / CLI / REST with a missing cache_dir fail immediately
# instead of hanging on a 600s provider timeout. Live-recording paths opt in.
FAIRPIPE_LLM_FORBID_LIVE = "FAIRPIPE_LLM_FORBID_LIVE"
FAIRPIPE_LLM_ALLOW_LIVE = "FAIRPIPE_LLM_ALLOW_LIVE"

_TRUTHY_ENV = ("1", "true", "yes", "on")
_FALSEY_ENV = ("0", "false", "no", "off")

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_SECONDS = 0.05
DEFAULT_BATCH_CONCURRENCY = 8


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM completion client with provider-specific backends."""

    provider: str
    model: str

    def available(self) -> bool:
        """Return True when the SDK is importable and credentials are present (if required)."""
        ...

    async def complete(self, prompt: str, *, params: Optional[Dict[str, Any]] = None) -> str:
        """Complete a single prompt."""
        ...

    async def complete_batch(
        self,
        prompts: List[str],
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Complete multiple prompts concurrently."""
        ...


class _BaseLLMClient(ABC):
    provider: str
    model: str

    def __init__(
        self,
        model: str,
        *,
        cache: Optional[ResponseCache] = None,
        replay_only: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    ) -> None:
        self.model = model
        self.cache = cache
        self.replay_only = replay_only
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.batch_concurrency = batch_concurrency

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def _complete_uncached(
        self, prompt: str, *, params: Optional[Dict[str, Any]] = None
    ) -> str:
        raise NotImplementedError

    async def _complete_with_retry(
        self, prompt: str, *, params: Optional[Dict[str, Any]] = None
    ) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._complete_uncached(prompt, params=params)
            except RateLimitError as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_seconds * (2**attempt)
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def complete(self, prompt: str, *, params: Optional[Dict[str, Any]] = None) -> str:
        merged = dict(params or {})
        key = make_cache_key(self.provider, self.model, prompt, merged)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            if self.replay_only:
                raise CacheMissError(
                    "Cache miss in replay-only mode; live provider calls are disabled.",
                    context={"cache_key": key, "provider": self.provider, "model": self.model},
                )
        response = await self._complete_with_retry(prompt, params=merged)
        if self.cache is not None:
            self.cache.set(key, response)
        return response

    async def complete_batch(
        self,
        prompts: List[str],
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        sem = asyncio.Semaphore(self.batch_concurrency)

        async def _one(prompt: str) -> str:
            async with sem:
                return await self.complete(prompt, params=params)

        return await asyncio.gather(*[_one(p) for p in prompts])


class CacheMissError(FairnessToolkitError):
    """Raised when replay-only mode encounters a cache miss."""


class LiveLLMCallForbidden(FairnessToolkitError):
    """Raised when a live SDK call is attempted without an explicit opt-in."""


class RateLimitError(Exception):
    """Raised when the provider returns HTTP 429 / rate-limit response."""


def live_llm_calls_forbidden() -> bool:
    """Live provider HTTP is forbidden unless explicitly opted in.

    Opt in with ``FAIRPIPE_LLM_ALLOW_LIVE=1`` (populate helpers and
    ``@pytest.mark.live_llm`` do this). ``FAIRPIPE_LLM_FORBID_LIVE=0`` also
    allows. Unset env — including a Jupyter kernel that never saw pytest —
    forbids, so a missing ``cache_dir`` fails immediately instead of hanging.
    """
    allow = os.getenv(FAIRPIPE_LLM_ALLOW_LIVE, "")
    if allow.lower() in _TRUTHY_ENV:
        return False
    forbid = os.getenv(FAIRPIPE_LLM_FORBID_LIVE)
    if forbid is not None and forbid.lower() in _FALSEY_ENV:
        return False
    return True


@contextmanager
def allow_live_llm_calls() -> Iterator[None]:
    """Opt this process into live provider HTTP for the duration of the block."""
    previous = os.environ.get(FAIRPIPE_LLM_ALLOW_LIVE)
    os.environ[FAIRPIPE_LLM_ALLOW_LIVE] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FAIRPIPE_LLM_ALLOW_LIVE, None)
        else:
            os.environ[FAIRPIPE_LLM_ALLOW_LIVE] = previous


def raise_if_live_llm_forbidden() -> None:
    """Raise immediately at the SDK call site unless live calls were opted in."""
    if not live_llm_calls_forbidden():
        return
    raise LiveLLMCallForbidden(
        "Live LLM provider call blocked. Live HTTP is forbidden by default so a "
        "missing or misconfigured cache_dir fails immediately instead of hanging "
        f"(Jupyter, CLI, REST, pytest). Set {FAIRPIPE_LLM_ALLOW_LIVE}=1 to opt in "
        "(live recording, @pytest.mark.live_llm, or a server that should bill the "
        "shared provider key).",
        suggestion=(
            "Point cache_dir at a recorded cache, or set "
            f"{FAIRPIPE_LLM_ALLOW_LIVE}=1 for an intentional live call."
        ),
    )


class OpenAICompatibleClient(_BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible HTTP APIs."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        cache: Optional[ResponseCache] = None,
        replay_only: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    ) -> None:
        super().__init__(
            model,
            cache=cache,
            replay_only=replay_only,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            batch_concurrency=batch_concurrency,
        )
        self._api_key = api_key if api_key is not None else os.getenv(OPENAI_API_KEY_ENV)
        self._base_url = base_url
        self._client = client
        self._sdk_ok = False
        if client is not None:
            self._sdk_ok = True
        else:
            try:
                import openai  # noqa: F401

                self._sdk_ok = True
            except ImportError:
                self._sdk_ok = False

    def available(self) -> bool:
        return bool(self._sdk_ok and self._api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise DependencyError(
                "openai package is required for provider='openai'.",
                dependency_name="openai",
                extra_name="llm",
            ) from exc
        kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncOpenAI(**kwargs)

    async def _complete_uncached(
        self, prompt: str, *, params: Optional[Dict[str, Any]] = None
    ) -> str:
        # Injected test doubles are not live SDK calls; skip the kill-switch.
        if self._client is None:
            raise_if_live_llm_forbidden()
        if not self.available():
            raise DependencyError(
                "OpenAI client is not available (missing SDK or OPENAI_API_KEY).",
                dependency_name="openai",
                extra_name="llm",
            )
        client = self._get_client()
        request_params = dict(params or {})
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **request_params,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            raise
        return _extract_openai_text(response)


class AnthropicClient(_BaseLLMClient):
    """Client for Anthropic's Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        cache: Optional[ResponseCache] = None,
        replay_only: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    ) -> None:
        super().__init__(
            model,
            cache=cache,
            replay_only=replay_only,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            batch_concurrency=batch_concurrency,
        )
        self._api_key = api_key if api_key is not None else os.getenv(ANTHROPIC_API_KEY_ENV)
        self._client = client
        self._sdk_ok = False
        if client is not None:
            self._sdk_ok = True
        else:
            try:
                import anthropic  # noqa: F401

                self._sdk_ok = True
            except ImportError:
                self._sdk_ok = False

    def available(self) -> bool:
        return bool(self._sdk_ok and self._api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise DependencyError(
                "anthropic package is required for provider='anthropic'.",
                dependency_name="anthropic",
                extra_name="llm",
            ) from exc
        return AsyncAnthropic(api_key=self._api_key)

    async def _complete_uncached(
        self, prompt: str, *, params: Optional[Dict[str, Any]] = None
    ) -> str:
        if self._client is None:
            raise_if_live_llm_forbidden()
        if not self.available():
            raise DependencyError(
                "Anthropic client is not available (missing SDK or ANTHROPIC_API_KEY).",
                dependency_name="anthropic",
                extra_name="llm",
            )
        client = self._get_client()
        request_params = dict(params or {})
        max_tokens = request_params.pop("max_tokens", 256)
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **request_params,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            raise
        return _extract_anthropic_text(response)


class LocalLLMClient(_BaseLLMClient):
    """Stub local/self-hosted provider — no credentials required (Phase 0 scaffolding)."""

    provider = "local"

    def __init__(
        self,
        model: str,
        *,
        responder: Optional[Any] = None,
        cache: Optional[ResponseCache] = None,
        replay_only: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    ) -> None:
        super().__init__(
            model,
            cache=cache,
            replay_only=replay_only,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            batch_concurrency=batch_concurrency,
        )
        self._responder = responder

    def available(self) -> bool:
        return True

    async def _complete_uncached(
        self, prompt: str, *, params: Optional[Dict[str, Any]] = None
    ) -> str:
        if self._responder is not None:
            result = self._responder(prompt, params=params or {})
            if asyncio.iscoroutine(result):
                return await result
            return str(result)
        return f"[local:{self.model}] {prompt}"


def get_llm_client(
    provider: str,
    model: str,
    *,
    cache: Optional[ResponseCache] = None,
    replay_only: bool = False,
    **kwargs: Any,
) -> LLMClient:
    """Factory for provider-specific LLM clients."""
    normalized = provider.strip().lower()
    client_kwargs = {**kwargs, "cache": cache, "replay_only": replay_only}
    if normalized == "openai":
        return OpenAICompatibleClient(model, **client_kwargs)
    if normalized == "anthropic":
        return AnthropicClient(model, **client_kwargs)
    if normalized == "local":
        return LocalLLMClient(model, **client_kwargs)
    raise ValueError(
        f"Unknown provider {provider!r}. Valid providers: 'openai', 'anthropic', 'local'."
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return "429" in message or "rate limit" in message


def _extract_openai_text(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return str(content or "")


def _extract_anthropic_text(response: Any) -> str:
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            return str(text)
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    return ""
