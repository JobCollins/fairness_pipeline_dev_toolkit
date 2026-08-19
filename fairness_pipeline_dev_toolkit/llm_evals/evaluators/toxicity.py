from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

from .._async_utils import run_coroutine
from ..client import LLMClient
from ..config import LLMEvalConfig
from ..guards import DEFAULT_LLM_MIN_GROUP_SIZE, apply_min_group_size
from ..probes.counterfactual import generate_counterfactual_prompts
from ..provenance import with_fixture_caveat
from ..scoring import (
    apply_scorer,
    bootstrap_rate_disparity,
    n_per_group,
    rate_disparity,
    toxicity_score,
)


class ToxicitySentimentEvaluator:
    """Max–min lexical toxicity/sentiment rate across demographic groups."""

    name = "toxicity_sentiment_disparity"

    def __init__(self, config: LLMEvalConfig, client: LLMClient) -> None:
        self.config = config
        self.client = client

    async def run_async(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
        scorer: Callable[[str], float] | None = None,
    ) -> tuple[MetricResult, List[Dict[str, str]]]:
        if self.config.counterfactual is None:
            raise ValueError("counterfactual config is required for toxicity_sentiment_disparity.")
        score_fn = scorer or toxicity_score
        prompts = generate_counterfactual_prompts(
            self.config.counterfactual.template,
            self.config.counterfactual.dimensions,
            self.config.counterfactual.defaults,
        )
        texts = await self.client.complete_batch(
            [item.prompt for item in prompts],
            params=self.config.params,
        )
        rows = [
            {
                "dimension": item.dimension,
                "group": item.group,
                "prompt": item.prompt,
                "response": text,
            }
            for item, text in zip(prompts, texts)
        ]
        counts = n_per_group(apply_scorer(rows, score_fn))
        eligible, can_compute = apply_min_group_size(
            counts, min_group_size, allow_small_samples=allow_small_samples
        )
        if not can_compute:
            return (
                with_fixture_caveat(
                    MetricResult(
                        metric="toxicity_sentiment_disparity",
                        value=float("nan"),
                        ci=None,
                        effect_size=float("nan"),
                        n_per_group=eligible,
                    ),
                    self.config.cache_dir,
                ),
                rows,
            )
        keep = set(eligible) if not allow_small_samples else set(counts)
        scores = apply_scorer([r for r in rows if r["group"] in keep], score_fn)
        value = rate_disparity(scores)
        ci = None
        if with_ci and np.isfinite(value):
            ci = bootstrap_rate_disparity(
                scores, B=bootstrap_B, level=ci_level, random_state=random_state
            )
        reporting = counts if allow_small_samples else eligible
        return (
            with_fixture_caveat(
                MetricResult(
                    metric="toxicity_sentiment_disparity",
                    value=float(value),
                    ci=ci,
                    effect_size=float(value) if np.isfinite(value) else float("nan"),
                    n_per_group=reporting,
                ),
                self.config.cache_dir,
            ),
            rows,
        )

    def toxicity_sentiment_disparity(self, **kwargs: Any) -> MetricResult:
        return run_coroutine(self.run_async(**kwargs))[0]
