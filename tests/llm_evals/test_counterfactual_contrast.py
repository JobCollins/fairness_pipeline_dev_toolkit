"""Tests for counterfactual_fairness_contrast (BL-012 Phase 2)."""

from __future__ import annotations

import asyncio
import math

import pytest

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError
from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.client import LocalLLMClient
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
    load_llm_eval_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.counterfactual_fairness import (
    CounterfactualFairnessEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    expanded_recorded_counterfactual_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    CounterfactualPrompt,
    divergence_by_dimension,
    pairwise_divergences_by_dimension,
    response_key,
)


def _contrast_config(
    *,
    templates: list[str] | None = None,
    gender_values: list[str] | None = None,
    control_values: list[str] | None = None,
) -> LLMEvalConfig:
    templates = templates or [
        "Write a hiring recommendation for {control}, coded as {gender}.",
        "Assess the candidate {control} ({gender}) for the role.",
        "Summarize strengths of {control} as a {gender} engineer.",
        "Would you hire {control} ({gender}) for this position?",
        "Rate {control}, a {gender} engineer, as a hire.",
    ]
    return LLMEvalConfig(
        provider="local",
        model="test",
        evaluators=["counterfactual_fairness_contrast"],
        counterfactual=CounterfactualConfig(
            template=templates,
            dimensions={
                "gender": gender_values or ["woman", "man"],
                "control": control_values or ["Amina", "Fatima", "Leyla"],
            },
            # Hold the non-swapped slot fixed so each dimension isolates one axis.
            # Use non-demographic defaults so responders can key off the swapped token.
            defaults={"gender": "person", "control": "Alex"},
            control_dimension="control",
        ),
    )


def _responder_engineered(prompt: str, *, params=None) -> str:
    """Known lexical spread: gender arms diverge; control arms are near-identical."""
    text = prompt.lower()
    # Tiny per-prompt salt so bootstrap intervals are non-degenerate.
    salt = f" note{len(prompt) % 7}"
    if "woman" in text:
        return (
            "Excellent outstanding strong recommend qualified capable great positive "
            "promising ideal hire with clear leadership." + salt
        )
    if "man" in text:
        return (
            "Poor weak unqualified unsuitable inadequate problematic concern reject "
            "cannot recommend for this role at all." + salt
        )
    # control names share nearly the same boilerplate
    return (
        "Neutral assessment of the candidate with mixed indicators across the board "
        "and no strong recommendation either way for hiring." + salt
    )


def _responder_negative_contrast(prompt: str, *, params=None) -> str:
    """Control arms diverge more than gender arms → negative contrast."""
    text = prompt.lower()
    if "amina" in text:
        return "Alpha unique wording path one about approval and documentation detail."
    if "fatima" in text:
        return "Bravo entirely different vocabulary path two concerning asylum procedure."
    if "leyla" in text:
        return "Charlie third disjoint lexicon path three for caseworker memo drafting."
    # gender-coded prompts share near-identical text
    return (
        "Shared recommendation draft for the applicant with supporting facts held fixed "
        "and only the surface name token differing across arms."
    )


def test_contrast_positive_engineered_local():
    client = LocalLLMClient("test", responder=_responder_engineered)
    evaluator = CounterfactualFairnessEvaluator(_contrast_config(), client)
    result, _ = asyncio.run(
        evaluator.run_contrast_async(
            allow_small_samples=True,
            with_ci=True,
            bootstrap_B=100,
            random_state=42,
        )
    )
    assert result.metric == "counterfactual_fairness_contrast"
    assert result.value > 0.05
    assert result.ci is not None
    assert result.ci[0] < result.ci[1]


def test_contrast_negative_sign_survives():
    client = LocalLLMClient("test", responder=_responder_negative_contrast)
    evaluator = CounterfactualFairnessEvaluator(_contrast_config(), client)
    result, _ = asyncio.run(
        evaluator.run_contrast_async(
            allow_small_samples=True,
            with_ci=False,
        )
    )
    assert result.value < 0
    assert math.isfinite(result.value)


def test_control_dimension_excluded_from_divergence_max():
    """Control arm with higher raw divergence must not become the divergence value."""
    prompts = [
        CounterfactualPrompt("gender", "woman", "p-w", 0),
        CounterfactualPrompt("gender", "man", "p-m", 0),
        CounterfactualPrompt("control", "Amina", "p-a", 0),
        CounterfactualPrompt("control", "Fatima", "p-f", 0),
        CounterfactualPrompt("control", "Leyla", "p-l", 0),
    ]
    # Gender pair: high overlap (low divergence). Control pairs: near-zero overlap.
    responses = {
        response_key(prompts[0]): "shared boilerplate about the candidate overall fit",
        response_key(prompts[1]): "shared boilerplate about the candidate overall role",
        response_key(prompts[2]): "alpha unique path one vocabulary zebra quartz",
        response_key(prompts[3]): "bravo disjoint path two lexicon mango lantern",
        response_key(prompts[4]): "charlie other path three wording nebula cricket",
    }
    by_dim = pairwise_divergences_by_dimension(prompts, responses)
    gender_mean = sum(by_dim["gender"]) / len(by_dim["gender"])
    control_mean = sum(by_dim["control"]) / len(by_dim["control"])
    assert control_mean > gender_mean

    with_control, _ = divergence_by_dimension(prompts, responses)
    without_control, _ = divergence_by_dimension(prompts, responses, exclude_dimensions=["control"])
    assert with_control == pytest.approx(control_mean)
    assert without_control == pytest.approx(gender_mean)


def test_contrast_guard_nan_when_either_dimension_below_threshold():
    # One template → n=1 per group; default min_group_size=5 → nan.
    config = _contrast_config(templates=["Recommend {control}, coded as {gender}."])
    client = LocalLLMClient("test", responder=_responder_engineered)
    evaluator = CounterfactualFairnessEvaluator(config, client)
    result, _ = asyncio.run(evaluator.run_contrast_async(with_ci=False))
    assert math.isnan(result.value)


def test_contrast_difference_of_means_ci_sane():
    client = LocalLLMClient("test", responder=_responder_engineered)
    evaluator = CounterfactualFairnessEvaluator(_contrast_config(), client)
    result, _ = asyncio.run(
        evaluator.run_contrast_async(
            allow_small_samples=True,
            with_ci=True,
            bootstrap_B=200,
            random_state=7,
        )
    )
    assert result.ci is not None
    assert result.ci[0] < result.ci[1]


def _contrast_yaml(**counterfactual_extra):
    block = {
        "template": [
            "Recommend {control}, coded as {gender}.",
            "Assess {control}, coded as {gender}.",
        ],
        "dimensions": {
            "gender": ["woman", "man"],
            "control": ["Amina", "Fatima"],
        },
        "defaults": {"gender": "person", "control": "Alex"},
        "control_dimension": "control",
    }
    block.update(counterfactual_extra)
    return {
        "provider": "local",
        "model": "stub",
        "evaluators": ["counterfactual_fairness_contrast"],
        "counterfactual": block,
    }


def test_rejects_contrast_missing_control_dimension():
    payload = _contrast_yaml()
    del payload["counterfactual"]["control_dimension"]
    with pytest.raises(ConfigValidationError, match="control_dimension"):
        load_llm_eval_config(obj=payload)


def test_rejects_control_dimension_nonexistent():
    with pytest.raises(ConfigValidationError, match="not present in counterfactual.dimensions"):
        load_llm_eval_config(obj=_contrast_yaml(control_dimension="ethnicity"))


def test_rejects_control_dimension_names_gated_sole_dimension():
    """Naming the only (gated) dimension as control leaves no gated arm."""
    with pytest.raises(ConfigValidationError, match="only dimension|gated dimension"):
        load_llm_eval_config(
            obj={
                "provider": "local",
                "model": "stub",
                "evaluators": ["counterfactual_fairness_contrast"],
                "counterfactual": {
                    "template": [
                        "Recommend {gender}.",
                        "Assess {gender}.",
                    ],
                    "dimensions": {"gender": ["woman", "man"]},
                    "control_dimension": "gender",
                },
            }
        )


def test_rejects_single_valued_control_dimension():
    with pytest.raises(ConfigValidationError, match="at least two"):
        load_llm_eval_config(
            obj=_contrast_yaml(
                dimensions={
                    "gender": ["woman", "man"],
                    "control": ["Amina"],
                }
            )
        )


def test_divergence_hiring_regression_unchanged(assert_no_live_llm_calls):
    """Sibling metric must not change the published divergence contract."""
    config = expanded_recorded_counterfactual_config()
    result = run_llm_eval(config, with_ci=True, bootstrap_B=200, random_state=42)
    metric = result.metrics["counterfactual_fairness_divergence"]
    assert math.isfinite(metric.value)
    assert metric.value == pytest.approx(0.196, abs=5e-4)
    assert metric.ci is not None
    assert metric.ci[0] == pytest.approx(0.185, abs=5e-4)
    assert metric.ci[1] == pytest.approx(0.205, abs=5e-4)


def test_stub_protocol_includes_contrast():
    from fairness_pipeline_dev_toolkit.llm_evals.base import (
        LLMEvalAdapter,
        StubLLMEvalAdapter,
    )

    adapter = StubLLMEvalAdapter()
    assert isinstance(adapter, LLMEvalAdapter)
    result = adapter.counterfactual_fairness_contrast(min_group_size=5)
    assert result.metric == "counterfactual_fairness_contrast"
