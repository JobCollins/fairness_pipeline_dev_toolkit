from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci

from .._async_utils import run_coroutine
from ..client import LLMClient
from ..config import CounterfactualConfig, LLMEvalConfig
from ..guards import (
    DEFAULT_LLM_MIN_GROUP_SIZE,
    apply_min_group_size,
    filter_items_by_eligible_groups,
)
from ..probes.counterfactual import (
    divergence_by_dimension,
    generate_counterfactual_prompts,
    matched_pairwise_divergences,
    response_key,
)


class CounterfactualFairnessEvaluator:
    """Counterfactual fairness probe — Phase 1 flagship evaluator."""

    name = "counterfactual_fairness"

    def __init__(
        self,
        config: LLMEvalConfig,
        client: LLMClient,
        *,
        counterfactual: Optional[CounterfactualConfig] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.counterfactual = counterfactual or config.counterfactual

    def available(self) -> bool:
        return self.client.available() if self.config.provider != "local" else True

    async def _collect_responses(
        self,
        prompts: List,
    ) -> Dict[Tuple[str, str, int], str]:
        responses: Dict[Tuple[str, str, int], str] = {}
        texts = await self.client.complete_batch(
            [item.prompt for item in prompts],
            params=self.config.params,
        )
        for item, text in zip(prompts, texts):
            responses[response_key(item)] = text
        return responses

    def _pairwise_divergences(
        self,
        prompts: List,
        responses: Dict[Tuple[str, str, int], str],
    ) -> List[float]:
        return matched_pairwise_divergences(prompts, responses)

    async def run_async(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
    ) -> tuple[MetricResult, List[Dict[str, str]]]:
        if self.counterfactual is None:
            raise ValueError("counterfactual config is required.")

        prompts = generate_counterfactual_prompts(
            self.counterfactual.template,
            self.counterfactual.dimensions,
            self.counterfactual.defaults,
        )
        if self.config.max_requests_per_run is not None:
            if len(prompts) > self.config.max_requests_per_run:
                raise ValueError(
                    f"Counterfactual probe requires {len(prompts)} requests but "
                    f"max_requests_per_run={self.config.max_requests_per_run}."
                )

        responses = await self._collect_responses(prompts)
        n_per_group: Dict[str, int] = {}
        for item in prompts:
            n_per_group[item.group] = n_per_group.get(item.group, 0) + 1

        eligible_n_per_group, can_compute = apply_min_group_size(
            n_per_group,
            min_group_size,
            allow_small_samples=allow_small_samples,
        )

        transcript_rows = [
            {
                "dimension": p.dimension,
                "group": p.group,
                "prompt": p.prompt,
                "response": responses[response_key(p)],
            }
            for p in prompts
        ]

        if not can_compute:
            return (
                MetricResult(
                    metric="counterfactual_fairness_divergence",
                    value=float("nan"),
                    ci=None,
                    effect_size=float("nan"),
                    n_per_group=eligible_n_per_group,
                ),
                transcript_rows,
            )

        if allow_small_samples:
            analysis_prompts = prompts
            reporting_n_per_group = n_per_group
        else:
            analysis_prompts = filter_items_by_eligible_groups(
                prompts, group_attr="group", eligible_groups=eligible_n_per_group
            )
            reporting_n_per_group = eligible_n_per_group

        value, _ = divergence_by_dimension(analysis_prompts, responses)

        ci = None
        pair_values = self._pairwise_divergences(analysis_prompts, responses)
        if with_ci and len(pair_values) >= 2 and np.isfinite(value):
            ci = bootstrap_ci(
                np.asarray(pair_values, dtype=float),
                stat_fn=np.mean,
                B=bootstrap_B,
                level=ci_level,
                random_state=random_state,
            )

        result = MetricResult(
            metric="counterfactual_fairness_divergence",
            value=float(value),
            ci=ci,
            effect_size=float(value) if np.isfinite(value) else float("nan"),
            n_per_group=reporting_n_per_group,
        )
        return result, transcript_rows

    def counterfactual_fairness_divergence(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
        **kwargs: Any,
    ) -> MetricResult:
        return run_coroutine(
            self.run_async(
                min_group_size=min_group_size,
                allow_small_samples=allow_small_samples,
                with_ci=with_ci,
                ci_level=ci_level,
                bootstrap_B=bootstrap_B,
                random_state=random_state,
            )
        )[0]

    def refusal_rate_disparity(self, **kwargs: Any) -> MetricResult:
        from .refusal import RefusalRateEvaluator

        return RefusalRateEvaluator(self.config, self.client).refusal_rate_disparity(**kwargs)

    def toxicity_sentiment_disparity(self, **kwargs: Any) -> MetricResult:
        from .toxicity import ToxicitySentimentEvaluator

        return ToxicitySentimentEvaluator(self.config, self.client).toxicity_sentiment_disparity(
            **kwargs
        )

    def stereotype_association_score(self, **kwargs: Any) -> MetricResult:
        from .stereotype import StereotypeAssociationEvaluator

        return StereotypeAssociationEvaluator(
            self.config, self.client
        ).stereotype_association_score(**kwargs)
