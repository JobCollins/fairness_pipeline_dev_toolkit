from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.stats.template_intervals import (
    T_MIN_TEMPLATES,
    small_template_note,
    template_bonferroni_t_max_mean,
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
    n_per_group_by_dimension,
    pairwise_divergences_by_dimension,
    response_key,
)
from ..provenance import with_fixture_caveat
from ..template_ci import contrast_template_arms, divergence_template_arms

#: Default when ``with_ci`` and ``ci_method`` is unset. Real-data coverage for
#: Bonferroni-t on per-template means did not meet the calibration criterion
#: (issue #63; see ``investigations/wave3a/``). Opt in with
#: ``ci_method="template_bonferroni_t"``.
C2B_UNCALIBRATED_NOTE = (
    "undefined:no_calibrated_interval (Bonferroni-t on per-template means did not "
    "meet the calibration criterion — mean coverage ≥ 0.95 and worst ≥ 0.93 for a "
    "95% interval — on real data at any template count; see "
    "https://github.com/JobCollins/fairness_pipeline_dev_toolkit/issues/63 and "
    "investigations/wave3a/)"
)


def _apply_c2b(arms, T: int, *, level: float, value: float) -> tuple:
    """Return (ci, ci_kind, ci_note) for template Bonferroni-t, or undefined."""
    if T < T_MIN_TEMPLATES:
        return (
            None,
            None,
            f"undefined:too_few_templates (T={T} < {T_MIN_TEMPLATES})",
        )
    try:
        lo, hi = template_bonferroni_t_max_mean(arms, level=level)
    except IntervalUndefinedError as err:
        return None, None, err.ci_note
    # Containment of the reported point (same hull idea as classifier gaps).
    lo, hi = min(lo, value), max(hi, value)
    return (float(lo), float(hi)), "template_bonferroni_t", small_template_note(T)


def _resolve_llm_ci(
    *,
    with_ci: bool,
    ci_method: Optional[str],
    arms,
    T: int,
    level: float,
    value: float,
    bootstrap_B: int,
) -> tuple:
    """Default CI is undefined; template Bonferroni-t is an explicit opt-in."""
    _ = bootstrap_B  # accepted for API compatibility; method is analytic
    if not with_ci:
        return None, None, None
    if ci_method is not None and ci_method not in ("template_bonferroni_t",):
        raise ValueError(
            f"Unknown ci_method {ci_method!r}; expected None or 'template_bonferroni_t'"
        )
    if ci_method is None:
        return None, None, C2B_UNCALIBRATED_NOTE
    return _apply_c2b(arms, T, level=level, value=value)


class DemographicSwapEvaluator:
    """Lexical divergence on demographically swapped prompts (Phase 1).

    Matched-template pairwise feature distance between name-/group-swapped
    prompts — a perturbation / invariance test, not Kusner et al. causal
    counterfactual fairness. Formerly named ``CounterfactualFairnessEvaluator``;
    see BL-012 (no-effect baseline).
    """

    name = "demographic_swap"

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
            control_dimension=self.counterfactual.control_dimension,
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
        ci_method: Optional[str] = None,
    ) -> MetricResult:
        _ = random_state
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
                    metric="demographic_swap_divergence",
                    value=float("nan"),
                    ci=None,
                    effect_size=float("nan"),
                    n_per_group=eligible_n_per_group,
                    ci_note=C2B_UNCALIBRATED_NOTE if with_ci else None,
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

        dims = sorted(
            {p.dimension for p in analysis_prompts if exclude is None or p.dimension not in exclude}
        )
        ci = ci_kind = ci_note = None
        if with_ci and np.isfinite(value):
            arms, T = divergence_template_arms(analysis_prompts, responses, dimensions=dims)
            ci, ci_kind, ci_note = _resolve_llm_ci(
                with_ci=with_ci,
                ci_method=ci_method,
                arms=arms,
                T=T,
                level=ci_level,
                value=float(value),
                bootstrap_B=bootstrap_B,
            )

        return with_fixture_caveat(
            MetricResult(
                metric="demographic_swap_divergence",
                value=float(value),
                ci=ci,
                effect_size=float(value) if np.isfinite(value) else float("nan"),
                n_per_group=reporting_n_per_group,
                p_value=None,
                ci_kind=ci_kind,
                ci_note=ci_note,
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
        ci_method: Optional[str] = None,
    ) -> MetricResult:
        _ = random_state
        if self.counterfactual is None:
            raise ValueError("counterfactual config is required.")
        control_dim = self.counterfactual.control_dimension
        if not control_dim:
            raise ValueError(
                "counterfactual.control_dimension is required for " "demographic_swap_contrast."
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
                    metric="demographic_swap_contrast",
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
                    metric="demographic_swap_contrast",
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

        ci = ci_kind = ci_note = None
        if with_ci and np.isfinite(contrast):
            arms, T = contrast_template_arms(
                analysis_prompts,
                responses,
                gated_dimensions=gated_dims,
                control_dimension=control_dim,
            )
            ci, ci_kind, ci_note = _resolve_llm_ci(
                with_ci=with_ci,
                ci_method=ci_method,
                arms=arms,
                T=T,
                level=ci_level,
                value=contrast,
                bootstrap_B=bootstrap_B,
            )

        return with_fixture_caveat(
            MetricResult(
                metric="demographic_swap_contrast",
                value=contrast,
                ci=ci,
                effect_size=contrast if np.isfinite(contrast) else float("nan"),
                n_per_group=reporting_n,
                p_value=None,
                ci_kind=ci_kind,
                ci_note=ci_note,
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
        ci_method: Optional[str] = None,
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
            ci_method=ci_method,
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
        ci_method: Optional[str] = None,
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
            ci_method=ci_method,
        )
        return result, transcript_rows

    def demographic_swap_divergence(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
        ci_method: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Max mean matched-template divergence across dimensions.

        Default CI is undefined (``ci=None`` +
        ``ci_note="undefined:no_calibrated_interval (...)"``): Bonferroni-t on
        per-template means did not meet the calibration criterion on real data
        (issue #63; see ``investigations/wave3a/``). Opt in with
        ``ci_method="template_bonferroni_t"`` (analytic; ``bootstrap_B``
        ignored; refuses below ``T_MIN_TEMPLATES=5``). ``p_value`` is always
        ``None``.
        """
        return run_coroutine(
            self.run_async(
                min_group_size=min_group_size,
                allow_small_samples=allow_small_samples,
                with_ci=with_ci,
                ci_level=ci_level,
                bootstrap_B=bootstrap_B,
                random_state=random_state,
                ci_method=ci_method,
            )
        )[0]

    def demographic_swap_contrast(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
        ci_method: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Signed gated−control contrast of matched-template divergences.

        Default CI is undefined (same as :meth:`demographic_swap_divergence`).
        Opt in with ``ci_method="template_bonferroni_t"``. ``p_value`` is always
        ``None``.
        """
        return run_coroutine(
            self.run_contrast_async(
                min_group_size=min_group_size,
                allow_small_samples=allow_small_samples,
                with_ci=with_ci,
                ci_level=ci_level,
                bootstrap_B=bootstrap_B,
                random_state=random_state,
                ci_method=ci_method,
            )
        )[0]

    def counterfactual_fairness_divergence(self, **kwargs: Any) -> MetricResult:
        """Deprecated alias for :meth:`demographic_swap_divergence`."""
        import warnings

        warnings.warn(
            "'counterfactual_fairness_divergence' is deprecated; use "
            "'demographic_swap_divergence' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.demographic_swap_divergence(**kwargs)

    def counterfactual_fairness_contrast(self, **kwargs: Any) -> MetricResult:
        """Deprecated alias for :meth:`demographic_swap_contrast`."""
        import warnings

        warnings.warn(
            "'counterfactual_fairness_contrast' is deprecated; use "
            "'demographic_swap_contrast' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.demographic_swap_contrast(**kwargs)

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
