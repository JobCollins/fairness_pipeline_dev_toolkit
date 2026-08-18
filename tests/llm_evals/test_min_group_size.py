"""Tests for shared LLM eval min_group_size guard (classifier-parity semantics)."""

from __future__ import annotations

import asyncio
import math

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.client import LocalLLMClient
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
)
from fairness_pipeline_dev_toolkit.llm_evals.demo import biased_hiring_responder
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.counterfactual_fairness import (
    CounterfactualFairnessEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.guards import (
    DEFAULT_LLM_MIN_GROUP_SIZE,
    apply_min_group_size,
)


def _config(*, gender_values=None) -> LLMEvalConfig:
    gender_values = gender_values or ["woman", "man", "nonbinary"]
    return LLMEvalConfig(
        provider="local",
        model="test",
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual=CounterfactualConfig(
            template="Write a hiring recommendation for {name}, a {gender} engineer.",
            dimensions={"gender": gender_values},
            defaults={"name": "Alex"},
        ),
    )


def test_apply_min_group_size_excludes_small_groups_like_native_adapter():
    eligible, ok = apply_min_group_size({"A": 1, "B": 4, "C": 6}, min_group_size=5)
    assert ok is False
    assert eligible == {"C": 6}


def test_apply_min_group_size_requires_two_eligible_groups():
    eligible, ok = apply_min_group_size({"A": 5, "B": 6}, min_group_size=5)
    assert ok is True
    assert eligible == {"A": 5, "B": 6}


def test_counterfactual_guard_returns_nan_below_threshold():
    """Explicit negative case: n=1 per group at default threshold → nan (classifier parity)."""
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_config(), client)

    result, _ = asyncio.run(
        evaluator.run_async(
            min_group_size=DEFAULT_LLM_MIN_GROUP_SIZE,
            allow_small_samples=False,
            with_ci=False,
        )
    )

    assert math.isnan(result.value)
    assert result.ci is None
    assert result.n_per_group == {}


def test_allow_small_samples_override_computes_illustrative_metric():
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_config(gender_values=["woman", "man"]), client)

    result, _ = asyncio.run(
        evaluator.run_async(
            min_group_size=DEFAULT_LLM_MIN_GROUP_SIZE,
            allow_small_samples=True,
            with_ci=False,
        )
    )

    assert result.value == pytest.approx(0.5972222222222222, rel=1e-3)
    assert result.n_per_group == {"woman": 1, "man": 1}
