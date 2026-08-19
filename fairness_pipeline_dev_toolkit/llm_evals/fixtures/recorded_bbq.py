from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fairness_pipeline_dev_toolkit.llm_evals.bbq import (
    DEFAULT_LOCAL_FIXTURE,
    item_to_prompt,
    load_bbq_items,
)
from fairness_pipeline_dev_toolkit.llm_evals.cache import ResponseCache, make_cache_key
from fairness_pipeline_dev_toolkit.llm_evals.client import get_llm_client
from fairness_pipeline_dev_toolkit.llm_evals.config import LLMEvalConfig
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_counterfactual import (
    RECORDED_MODEL,
    RECORDED_PARAMS,
    RECORDED_PROVIDER,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import CAVEAT_RECORDED_BBQ

_BBQ_ROOT = Path(__file__).resolve().parent / "recorded_bbq"
RECORDED_BBQ_CACHE_DIR = _BBQ_ROOT / "cache"
RECORDED_BBQ_MANIFEST_PATH = _BBQ_ROOT / "manifest.json"


def default_recorded_bbq_config() -> LLMEvalConfig:
    return LLMEvalConfig(
        provider=RECORDED_PROVIDER,
        model=RECORDED_MODEL,
        evaluators=["stereotype_association_score"],
        cache_dir=str(RECORDED_BBQ_CACHE_DIR),
        params=dict(RECORDED_PARAMS),
        bbq_path=str(DEFAULT_LOCAL_FIXTURE),
    )


async def populate_recorded_bbq_cache(
    *,
    provider: str = RECORDED_PROVIDER,
    model: str = RECORDED_MODEL,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    params = dict(params or RECORDED_PARAMS)
    items = load_bbq_items(DEFAULT_LOCAL_FIXTURE)
    cache = ResponseCache(RECORDED_BBQ_CACHE_DIR)
    client = get_llm_client(provider, model, cache=None)
    if not client.available():
        raise RuntimeError(
            f"Provider {provider!r} is not available. Set ANTHROPIC_API_KEY before recording."
        )
    recorded = []
    for item in items:
        prompt = item_to_prompt(item)
        key = make_cache_key(provider, model, prompt, params)
        response = await client.complete(prompt, params=params)
        cache.set(key, response)
        recorded.append(
            {
                "group": item.get("group"),
                "prompt": prompt,
                "cache_key": key,
                "response_preview": response[:120],
            }
        )
    manifest = {
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": provider,
        "model": model,
        "params": params,
        "bbq_path": str(DEFAULT_LOCAL_FIXTURE),
        "illustrative": True,
        "caveat": CAVEAT_RECORDED_BBQ,
        "prompts": recorded,
    }
    _BBQ_ROOT.mkdir(parents=True, exist_ok=True)
    RECORDED_BBQ_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def populate_recorded_bbq_cache_sync(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(populate_recorded_bbq_cache(**kwargs))
