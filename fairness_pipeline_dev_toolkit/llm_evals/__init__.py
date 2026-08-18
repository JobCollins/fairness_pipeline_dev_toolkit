"""
Public API for LLM fairness evaluation (Phase 0 scaffolding).

Evaluators and CLI integration ship in Phase 1+.
"""

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

from .base import LLMEvalAdapter, StubLLMEvalAdapter
from .cache import ResponseCache, make_cache_key
from .client import (
    AnthropicClient,
    CacheMissError,
    LLMClient,
    LocalLLMClient,
    OpenAICompatibleClient,
    RateLimitError,
    get_llm_client,
)
from .config import CounterfactualConfig, LLMEvalConfig, load_llm_eval_config
from .dry_run import DryRunEstimate, estimate_dry_run
from .evaluators import CounterfactualFairnessEvaluator
from .fixtures import (
    default_recorded_counterfactual_config,
    load_recorded_manifest,
    populate_recorded_counterfactual_cache,
)
from .guards import DEFAULT_LLM_MIN_GROUP_SIZE, apply_min_group_size
from .runner import (
    LLMEvalRunResult,
    results_to_markdown,
    run_llm_eval,
    write_transcripts,
)

__all__ = [
    "LLMEvalAdapter",
    "StubLLMEvalAdapter",
    "MetricResult",
    "LLMClient",
    "OpenAICompatibleClient",
    "AnthropicClient",
    "LocalLLMClient",
    "get_llm_client",
    "CacheMissError",
    "RateLimitError",
    "ResponseCache",
    "make_cache_key",
    "CounterfactualConfig",
    "LLMEvalConfig",
    "load_llm_eval_config",
    "CounterfactualFairnessEvaluator",
    "DryRunEstimate",
    "estimate_dry_run",
    "LLMEvalRunResult",
    "run_llm_eval",
    "results_to_markdown",
    "write_transcripts",
    "DEFAULT_LLM_MIN_GROUP_SIZE",
    "apply_min_group_size",
    "default_recorded_counterfactual_config",
    "load_recorded_manifest",
    "populate_recorded_counterfactual_cache",
]
