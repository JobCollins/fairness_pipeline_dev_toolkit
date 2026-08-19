from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from fairness_pipeline_dev_toolkit.integration.reporting import to_markdown_report
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

from ._async_utils import run_coroutine
from .bbq import load_bbq_items
from .cache import ResponseCache
from .client import LLMClient, get_llm_client
from .config import LLMEvalConfig, load_llm_eval_config
from .dry_run import DryRunEstimate, estimate_dry_run
from .evaluators.counterfactual_fairness import CounterfactualFairnessEvaluator
from .evaluators.refusal import RefusalRateEvaluator
from .evaluators.stereotype import StereotypeAssociationEvaluator
from .evaluators.toxicity import ToxicitySentimentEvaluator
from .guards import DEFAULT_LLM_MIN_GROUP_SIZE
from .probes.counterfactual import as_template_list


@dataclass
class LLMEvalRunResult:
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    dry_run: Optional[DryRunEstimate] = None
    transcripts: Dict[str, Any] = field(default_factory=dict)


def build_client(
    config: LLMEvalConfig,
    *,
    cache: Optional[ResponseCache] = None,
    replay_only: bool = False,
) -> LLMClient:
    if config.provider == "local":
        from .demo import build_local_demo_client

        return build_local_demo_client(config.model)
    return get_llm_client(
        config.provider,
        config.model,
        cache=cache,
        replay_only=replay_only,
    )


async def run_llm_eval_async(
    config: LLMEvalConfig,
    *,
    client: Optional[LLMClient] = None,
    dry_run: bool = False,
    min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
    allow_small_samples: Optional[bool] = None,
    with_ci: bool = True,
    ci_level: float = 0.95,
    bootstrap_B: int = 200,
    random_state: int = 42,
) -> LLMEvalRunResult:
    cache = ResponseCache(config.cache_dir) if config.cache_dir else None
    replay_only = bool(config.cache_dir)
    llm_client = client or build_client(config, cache=cache, replay_only=replay_only)
    allow_small = config.allow_small_samples if allow_small_samples is None else allow_small_samples

    cf_dims = config.counterfactual.dimensions if config.counterfactual else None
    n_templates = (
        len(as_template_list(config.counterfactual.template)) if config.counterfactual else 1
    )
    estimate = estimate_dry_run(
        provider=config.provider,
        model=config.model,
        evaluators=config.evaluators,
        counterfactual_dimensions=cf_dims,
        n_templates=n_templates,
        bbq_item_count=(
            len(load_bbq_items(config.bbq_path))
            if "stereotype_association_score" in config.evaluators
            else None
        ),
    )
    if dry_run:
        return LLMEvalRunResult(metrics={}, dry_run=estimate)

    metrics: Dict[str, MetricResult] = {}
    transcripts: Dict[str, Any] = {}

    if "counterfactual_fairness_divergence" in config.evaluators:
        evaluator = CounterfactualFairnessEvaluator(config, llm_client)
        metric, rows = await evaluator.run_async(
            min_group_size=min_group_size,
            allow_small_samples=allow_small,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
        )
        metrics["counterfactual_fairness_divergence"] = metric
        transcripts["counterfactual"] = rows

    if "refusal_rate_disparity" in config.evaluators:
        evaluator = RefusalRateEvaluator(config, llm_client)
        metric, rows = await evaluator.run_async(
            min_group_size=min_group_size,
            allow_small_samples=allow_small,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
        )
        metrics["refusal_rate_disparity"] = metric
        transcripts["refusal"] = rows

    if "toxicity_sentiment_disparity" in config.evaluators:
        evaluator = ToxicitySentimentEvaluator(config, llm_client)
        metric, rows = await evaluator.run_async(
            min_group_size=min_group_size,
            allow_small_samples=allow_small,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
        )
        metrics["toxicity_sentiment_disparity"] = metric
        transcripts["toxicity"] = rows

    if "stereotype_association_score" in config.evaluators:
        evaluator = StereotypeAssociationEvaluator(config, llm_client)
        metric, rows = await evaluator.run_async(
            min_group_size=min_group_size,
            allow_small_samples=allow_small,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
        )
        metrics["stereotype_association_score"] = metric
        transcripts["stereotype"] = rows

    return LLMEvalRunResult(metrics=metrics, dry_run=estimate, transcripts=transcripts)


def run_llm_eval(
    config: LLMEvalConfig | str,
    *,
    client: Optional[LLMClient] = None,
    dry_run: bool = False,
    min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
    allow_small_samples: Optional[bool] = None,
    with_ci: bool = True,
    ci_level: float = 0.95,
    bootstrap_B: int = 200,
    random_state: int = 42,
) -> LLMEvalRunResult:
    if isinstance(config, str):
        config = load_llm_eval_config(path=config)
    elif isinstance(config, dict):
        config = load_llm_eval_config(obj=config)
    return run_coroutine(
        run_llm_eval_async(
            config,
            client=client,
            dry_run=dry_run,
            min_group_size=min_group_size,
            allow_small_samples=allow_small_samples,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
        )
    )


def results_to_markdown(
    result: LLMEvalRunResult, *, title: str = "LLM Fairness Evaluation Report"
) -> str:
    if result.dry_run and not result.metrics:
        return result.dry_run.to_markdown()
    body = to_markdown_report(result.metrics, title=title)
    if result.dry_run is not None:
        body += (
            "\n\n## Request estimate\n\n"
            f"- Estimated requests: {result.dry_run.request_count}\n"
            f"- Estimated cost (USD): ${result.dry_run.estimated_cost_usd:.4f}\n"
        )
    return body


def write_transcripts(path: str | Path, transcripts: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(transcripts, indent=2), encoding="utf-8")
