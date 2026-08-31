from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def make_cache_key(
    provider: str,
    model: str,
    prompt: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a stable cache key from provider, model, prompt, and params."""
    payload = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "params": params or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """File-backed LLM response cache keyed on (provider, model, prompt, params)."""

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".fairpipe" / "llm_cache"
        self.cache_dir = Path(cache_dir).expanduser()
        self.hits = 0
        self.misses = 0

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.txt"

    def contains(self, key: str) -> bool:
        return self._path_for_key(key).exists()

    def get(self, key: str) -> Optional[str]:
        path = self._path_for_key(key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return path.read_text(encoding="utf-8")

    def set(self, key: str, response: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path_for_key(key).write_text(response, encoding="utf-8")

    def get_or_compute_key(
        self,
        provider: str,
        model: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the cache key for a completion request."""
        return make_cache_key(provider, model, prompt, params)
