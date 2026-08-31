"""Public API for LLM fairness evaluation."""

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

from .base import LLMEvalAdapter, StubLLMEvalAdapter
from .cache import ResponseCache, make_cache_key
from .client import (
    FAIRPIPE_LLM_ALLOW_LIVE,
    FAIRPIPE_LLM_FORBID_LIVE,
    AnthropicClient,
    CacheMissError,
    LiveLLMCallForbidden,
    LLMClient,
    LocalLLMClient,
    OpenAICompatibleClient,
    RateLimitError,
    allow_live_llm_calls,
    get_llm_client,
)
from .config import CounterfactualConfig, LLMEvalConfig, load_llm_eval_config
from .dry_run import DryRunEstimate, estimate_dry_run
from .evaluators import (
    CounterfactualFairnessEvaluator,
    RefusalRateEvaluator,
    StereotypeAssociationEvaluator,
    ToxicitySentimentEvaluator,
)
from .fixtures import (
    default_recorded_bbq_config,
    default_recorded_counterfactual_config,
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    expanded_recorded_counterfactual_config,
    load_recorded_manifest,
    populate_recorded_counterfactual_cache,
)
from .gating import (
    EXIT_FAIL,
    EXIT_ILLUSTRATIVE,
    EXIT_PASS,
    EXIT_USAGE,
    GATE_FAIL,
    GATE_ILLUSTRATIVE,
    GATE_PASS,
    GATE_STATUS_TO_EXIT,
    evaluate_llm_eval_gate,
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
    "LiveLLMCallForbidden",
    "FAIRPIPE_LLM_FORBID_LIVE",
    "FAIRPIPE_LLM_ALLOW_LIVE",
    "allow_live_llm_calls",
    "RateLimitError",
    "ResponseCache",
    "make_cache_key",
    "CounterfactualConfig",
    "LLMEvalConfig",
    "load_llm_eval_config",
    "CounterfactualFairnessEvaluator",
    "RefusalRateEvaluator",
    "ToxicitySentimentEvaluator",
    "StereotypeAssociationEvaluator",
    "DryRunEstimate",
    "estimate_dry_run",
    "LLMEvalRunResult",
    "run_llm_eval",
    "results_to_markdown",
    "write_transcripts",
    "DEFAULT_LLM_MIN_GROUP_SIZE",
    "apply_min_group_size",
    "evaluate_llm_eval_gate",
    "GATE_PASS",
    "GATE_FAIL",
    "GATE_ILLUSTRATIVE",
    "GATE_STATUS_TO_EXIT",
    "EXIT_PASS",
    "EXIT_FAIL",
    "EXIT_USAGE",
    "EXIT_ILLUSTRATIVE",
    "default_recorded_counterfactual_config",
    "expanded_recorded_counterfactual_config",
    "default_recorded_refusal_config",
    "default_recorded_toxicity_config",
    "default_recorded_bbq_config",
    "load_recorded_manifest",
    "populate_recorded_counterfactual_cache",
]
