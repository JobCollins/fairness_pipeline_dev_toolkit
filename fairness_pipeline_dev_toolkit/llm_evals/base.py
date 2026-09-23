from __future__ import annotations

import warnings
from typing import Any, Protocol, runtime_checkable

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

from .guards import DEFAULT_LLM_MIN_GROUP_SIZE
from .names import DEMOGRAPHIC_SWAP_CONTRAST, DEMOGRAPHIC_SWAP_DIVERGENCE


@runtime_checkable
class LLMEvalAdapter(Protocol):
    """Protocol for LLM fairness evaluators — sibling to MetricAdapter, not a subclass."""

    name: str

    def available(self) -> bool:
        """Return True if the provider SDK is installed and credentials are present."""
        ...

    def demographic_swap_divergence(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...

    def demographic_swap_contrast(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...

    def counterfactual_fairness_divergence(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...

    def counterfactual_fairness_contrast(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...

    def refusal_rate_disparity(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...

    def toxicity_sentiment_disparity(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...

    def stereotype_association_score(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult: ...


class StubLLMEvalAdapter:
    """Minimal concrete adapter for interface contract tests (Phase 0 scaffolding only)."""

    name = "stub"

    def available(self) -> bool:
        return True

    def demographic_swap_divergence(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        return MetricResult(
            metric=DEMOGRAPHIC_SWAP_DIVERGENCE,
            value=0.0,
            n_per_group={"A": min_group_size, "B": min_group_size},
        )

    def demographic_swap_contrast(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        return MetricResult(
            metric=DEMOGRAPHIC_SWAP_CONTRAST,
            value=0.0,
            n_per_group={"A": min_group_size, "B": min_group_size},
        )

    def counterfactual_fairness_divergence(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        warnings.warn(
            "'counterfactual_fairness_divergence' is deprecated; use "
            "'demographic_swap_divergence' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.demographic_swap_divergence(min_group_size=min_group_size, **kwargs)

    def counterfactual_fairness_contrast(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        warnings.warn(
            "'counterfactual_fairness_contrast' is deprecated; use "
            "'demographic_swap_contrast' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.demographic_swap_contrast(min_group_size=min_group_size, **kwargs)

    def refusal_rate_disparity(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        return MetricResult(
            metric="refusal_rate_disparity",
            value=0.0,
            n_per_group={"A": min_group_size, "B": min_group_size},
        )

    def toxicity_sentiment_disparity(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        return MetricResult(
            metric="toxicity_sentiment_disparity",
            value=0.0,
            n_per_group={"A": min_group_size, "B": min_group_size},
        )

    def stereotype_association_score(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        **kwargs: Any,
    ) -> MetricResult:
        return MetricResult(
            metric="stereotype_association_score",
            value=0.0,
            n_per_group={"A": min_group_size, "B": min_group_size},
        )
