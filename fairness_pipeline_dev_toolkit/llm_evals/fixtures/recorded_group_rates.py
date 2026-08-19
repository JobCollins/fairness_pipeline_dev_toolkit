from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fairness_pipeline_dev_toolkit.llm_evals.cache import make_cache_key
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_counterfactual import (
    EXPANDED_COUNTERFACTUAL_CACHE_DIR,
    EXPANDED_COUNTERFACTUAL_TEMPLATES,
    RECORDED_COUNTERFACTUAL_DEFAULTS,
    RECORDED_COUNTERFACTUAL_DIMENSIONS,
    RECORDED_MODEL,
    RECORDED_PARAMS,
    RECORDED_PROVIDER,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    generate_counterfactual_prompts,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import (
    CAVEAT_RECORDED_REFUSAL,
    CAVEAT_RECORDED_TOXICITY,
)

_REFUSAL_ROOT = Path(__file__).resolve().parent / "recorded_refusal"
RECORDED_REFUSAL_CACHE_DIR = _REFUSAL_ROOT / "cache"
RECORDED_REFUSAL_MANIFEST_PATH = _REFUSAL_ROOT / "manifest.json"

_TOXICITY_ROOT = Path(__file__).resolve().parent / "recorded_toxicity"
RECORDED_TOXICITY_CACHE_DIR = _TOXICITY_ROOT / "cache"
RECORDED_TOXICITY_MANIFEST_PATH = _TOXICITY_ROOT / "manifest.json"


def _group_config(evaluator: str, cache_dir: Path) -> LLMEvalConfig:
    return LLMEvalConfig(
        provider=RECORDED_PROVIDER,
        model=RECORDED_MODEL,
        evaluators=[evaluator],
        counterfactual=CounterfactualConfig(
            template=list(EXPANDED_COUNTERFACTUAL_TEMPLATES),
            dimensions=dict(RECORDED_COUNTERFACTUAL_DIMENSIONS),
            defaults=dict(RECORDED_COUNTERFACTUAL_DEFAULTS),
        ),
        cache_dir=str(cache_dir),
        params=dict(RECORDED_PARAMS),
    )


def default_recorded_refusal_config() -> LLMEvalConfig:
    return _group_config("refusal_rate_disparity", RECORDED_REFUSAL_CACHE_DIR)


def default_recorded_toxicity_config() -> LLMEvalConfig:
    return _group_config("toxicity_sentiment_disparity", RECORDED_TOXICITY_CACHE_DIR)


def _seed_from_expanded(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not EXPANDED_COUNTERFACTUAL_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Expanded counterfactual cache missing at {EXPANDED_COUNTERFACTUAL_CACHE_DIR}"
        )
    for src in EXPANDED_COUNTERFACTUAL_CACHE_DIR.glob("*.txt"):
        shutil.copy2(src, dest / src.name)


def _write_manifest(path: Path, evaluator: str, cache_dir: Path) -> Dict[str, Any]:
    config = _group_config(evaluator, cache_dir)
    assert config.counterfactual is not None
    prompts = generate_counterfactual_prompts(
        config.counterfactual.template,
        config.counterfactual.dimensions,
        config.counterfactual.defaults,
    )
    recorded = [
        {
            "dimension": item.dimension,
            "group": item.group,
            "prompt": item.prompt,
            "cache_key": make_cache_key(config.provider, config.model, item.prompt, config.params),
        }
        for item in prompts
    ]
    manifest = {
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": RECORDED_PROVIDER,
        "model": RECORDED_MODEL,
        "params": RECORDED_PARAMS,
        "evaluator": evaluator,
        "source_cache": str(EXPANDED_COUNTERFACTUAL_CACHE_DIR),
        "illustrative": True,
        "caveat": (
            CAVEAT_RECORDED_REFUSAL
            if evaluator == "refusal_rate_disparity"
            else CAVEAT_RECORDED_TOXICITY
        ),
        "prompts": recorded,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def populate_recorded_refusal_cache() -> Dict[str, Any]:
    """Copy expanded hiring-response cache (n=9/group) into the refusal fixture dir."""
    _seed_from_expanded(RECORDED_REFUSAL_CACHE_DIR)
    return _write_manifest(
        RECORDED_REFUSAL_MANIFEST_PATH, "refusal_rate_disparity", RECORDED_REFUSAL_CACHE_DIR
    )


def populate_recorded_toxicity_cache() -> Dict[str, Any]:
    """Copy expanded hiring-response cache (n=9/group) into the toxicity fixture dir."""
    _seed_from_expanded(RECORDED_TOXICITY_CACHE_DIR)
    return _write_manifest(
        RECORDED_TOXICITY_MANIFEST_PATH,
        "toxicity_sentiment_disparity",
        RECORDED_TOXICITY_CACHE_DIR,
    )
