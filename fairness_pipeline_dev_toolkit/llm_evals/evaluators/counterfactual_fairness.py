from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.stats.bootstrap import (
    bootstrap_ci,
    bootstrap_difference_of_means,
)

from .._async_utils import run_coroutine
from ..client import LLMClient
from ..config import CounterfactualConfig, LLMEvalConfig
from ..guards import (
    DEFAULT_LLM_MIN_GROUP_SIZE,
    apply_min_group_size,
    filter_items_by_eligible_groups,
)
from ..probes.counterfactual import (
    CounterfactualPrompt,
    divergence_by_dimension,
    generate_counterfactual_prompts,
    matched_pairwise_divergences,
    n_per_group_by_dimension,
    pairwise_divergences_by_dimension,
    response_key,
)
from ..provenance import with_fixture_caveat


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
        prompts: List[CounterfactualPrompt],
    ) -> Dict[Tuple[str, str, int], str]:
        responses: Dict[Tuple[str, str, int], str] = {}
        texts = await self.client.complete_batch(
            [item.prompt for item in prompts],
            params=self.config.params,
        )
        for item, text in zip(prompts, texts):
            responses[response_key(item)] = text
        return responses

    def _build_prompts(self) -> List[CounterfactualPrompt]:
        if self.counterfactual is None:
            raise ValueError("counterfactual config is required.")
        prompts = generate_counterfactual_prompts(
            self.counterfactual.template,
            self.counterfactual.dimensions,
            self.counterfactual.defaults,
            self.counterfactual.name_pools,
        )
        if self.config.max_requests_per_run is not None:
            if len(prompts) > self.config.max_requests_per_run:
                raise ValueError(
                    f"Counterfactual probe requires {len(prompts)} requests but "
                    f"max_requests_per_run={self.config.max_requests_per_run}."
                )
        return prompts

    @staticmethod
    def _transcript_rows(
        prompts: List[CounterfactualPrompt],
        responses: Dict[Tuple[str, str, int], str],
    ) -> List[Dict[str, str]]:
        return [
            {
                "dimension": p.dimension,
                "group": p.group,
                "prompt": p.prompt,
                "response": responses[response_key(p)],
            }
            for p in prompts
        ]

    def _exclude_dimensions(self) -> Optional[List[str]]:
        control = self.counterfactual.control_dimension if self.counterfactual else None
        return [control] if control else None

    def _compute_divergence(
        self,
        prompts: List[CounterfactualPrompt],
        responses: Dict[Tuple[str, str, int], str],
        *,
        min_group_size: int,
        allow_small_samples: bool,
        with_ci: bool,
        ci_level: float,
        bootstrap_B: int,
        random_state: int,
    ) -> MetricResult:
        n_per_group: Dict[str, int] = {}
        for item in prompts:
            n_per_group[item.group] = n_per_group.get(item.group, 0) + 1

        eligible_n_per_group, can_compute = apply_min_group_size(
            n_per_group,
            min_group_size,
            allow_small_samples=allow_small_samples,
        )

        if not can_compute:
            return with_fixture_caveat(
                MetricResult(
                    metric="counterfactual_fairness_divergence",
                    value=float("nan"),
                    ci=None,
                    effect_size=float("nan"),
                    n_per_group=eligible_n_per_group,
                ),
                self.config.cache_dir,
            )

        if allow_small_samples:
            analysis_prompts = prompts
            reporting_n_per_group = n_per_group
        else:
            analysis_prompts = filter_items_by_eligible_groups(
                prompts, group_attr="group", eligible_groups=eligible_n_per_group
            )
            reporting_n_per_group = eligible_n_per_group

        exclude = self._exclude_dimensions()
        value, _ = divergence_by_dimension(analysis_prompts, responses, exclude_dimensions=exclude)

        ci = None
        if exclude:
            gated_prompts = [p for p in analysis_prompts if p.dimension not in exclude]
            pair_values = matched_pairwise_divergences(gated_prompts, responses)
        else:
            pair_values = matched_pairwise_divergences(analysis_prompts, responses)
        if with_ci and len(pair_values) >= 2 and np.isfinite(value):
            ci = bootstrap_ci(
                np.asarray(pair_values, dtype=float),
                stat_fn=np.mean,
                B=bootstrap_B,
                level=ci_level,
                random_state=random_state,
            )

        return with_fixture_caveat(
            MetricResult(
                metric="counterfactual_fairness_divergence",
                value=float(value),
                ci=ci,
                effect_size=float(value) if np.isfinite(value) else float("nan"),
                n_per_group=reporting_n_per_group,
            ),
            self.config.cache_dir,
        )

    def _compute_contrast(
        self,
        prompts: List[CounterfactualPrompt],
        responses: Dict[Tuple[str, str, int], str],
        *,
        min_group_size: int,
        allow_small_samples: bool,
        with_ci: bool,
        ci_level: float,
        bootstrap_B: int,
        random_state: int,
    ) -> MetricResult:
        if self.counterfactual is None:
            raise ValueError("counterfactual config is required.")
        control_dim = self.counterfactual.control_dimension
        if not control_dim:
            raise ValueError(
                "counterfactual.control_dimension is required for "
                "counterfactual_fairness_contrast."
            )

        per_dim_counts = n_per_group_by_dimension(prompts)
        gated_dims = [d for d in self.counterfactual.dimensions if d != control_dim]

        reporting_n: Dict[str, int] = {}
        can_compute = True
        eligible_by_dim: Dict[str, Dict[str, int]] = {}
        for dim in [*gated_dims, control_dim]:
            counts = per_dim_counts.get(dim, {})
            eligible, ok = apply_min_group_size(
                counts,
                min_group_size,
                allow_small_samples=allow_small_samples,
            )
            eligible_by_dim[dim] = eligible
            reporting_n.update(eligible if not allow_small_samples else counts)
            if not ok:
                can_compute = False

        if not can_compute:
            return with_fixture_caveat(
                MetricResult(
                    metric="counterfactual_fairness_contrast",
                    value=float("nan"),
                    ci=None,
                    effect_size=float("nan"),
                    n_per_group=reporting_n,
                ),
                self.config.cache_dir,
            )

        if allow_small_samples:
            analysis_prompts = prompts
        else:
            eligible_groups = {group for eligible in eligible_by_dim.values() for group in eligible}
            analysis_prompts = [p for p in prompts if p.group in eligible_groups]

        by_dim = pairwise_divergences_by_dimension(analysis_prompts, responses)
        control_pairs = by_dim.get(control_dim, [])
        gated_means: List[float] = []
        gated_pairs: List[float] = []
        for dim in gated_dims:
            vals = by_dim.get(dim, [])
            if vals:
                gated_means.append(sum(vals) / len(vals))
                gated_pairs.extend(vals)

        if not gated_means or not control_pairs:
            return with_fixture_caveat(
                MetricResult(
                    metric="counterfactual_fairness_contrast",
                    value=float("nan"),
                    ci=None,
                    effect_size=float("nan"),
                    n_per_group=reporting_n,
                ),
                self.config.cache_dir,
            )

        gated_value = max(gated_means)
        control_value = sum(control_pairs) / len(control_pairs)
        # Signed: negative means cross-group ≤ within-group baseline (null reading).
        contrast = float(gated_value - control_value)

        ci = None
        if with_ci and len(gated_pairs) >= 2 and len(control_pairs) >= 2:
            ci = bootstrap_difference_of_means(
                np.asarray(gated_pairs, dtype=float),
                np.asarray(control_pairs, dtype=float),
                B=bootstrap_B,
                level=ci_level,
                random_state=random_state,
            )

        return with_fixture_caveat(
            MetricResult(
                metric="counterfactual_fairness_contrast",
                value=contrast,
                ci=ci,
                effect_size=contrast if np.isfinite(contrast) else float("nan"),
                n_per_group=reporting_n,
            ),
            self.config.cache_dir,
        )

    async def prepare_async(
        self,
    ) -> tuple[List[CounterfactualPrompt], Dict[Tuple[str, str, int], str], List[Dict[str, str]]]:
        """Generate prompts, collect responses, and build transcript rows once."""
        prompts = self._build_prompts()
        responses = await self._collect_responses(prompts)
        return prompts, responses, self._transcript_rows(prompts, responses)

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
        prompts, responses, transcript_rows = await self.prepare_async()
        result = self._compute_divergence(
            prompts,
            responses,
            min_group_size=min_group_size,
            allow_small_samples=allow_small_samples,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
        )
        return result, transcript_rows

    async def run_contrast_async(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
    ) -> tuple[MetricResult, List[Dict[str, str]]]:
        prompts, responses, transcript_rows = await self.prepare_async()
        result = self._compute_contrast(
            prompts,
            responses,
            min_group_size=min_group_size,
            allow_small_samples=allow_small_samples,
            with_ci=with_ci,
            ci_level=ci_level,
            bootstrap_B=bootstrap_B,
            random_state=random_state,
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

    def counterfactual_fairness_contrast(
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
            self.run_contrast_async(
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
