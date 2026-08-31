"""Unit tests for the counterfactual fairness probe."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

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


def test_counterfactual_probe_engineered_divergence_local():
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_config(gender_values=["woman", "man"]), client)

    result, _ = asyncio.run(
        evaluator.run_async(
            with_ci=False,
            bootstrap_B=50,
            random_state=42,
            allow_small_samples=True,
        )
    )

    assert result.metric == "counterfactual_fairness_divergence"
    assert result.value == pytest.approx(0.5972222222222222, rel=1e-3)
    assert result.ci is None


def test_counterfactual_probe_bootstrap_ci_matches_expected():
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_config(), client)

    result, _ = asyncio.run(
        evaluator.run_async(
            with_ci=True,
            bootstrap_B=200,
            random_state=42,
            allow_small_samples=True,
        )
    )

    assert result.value == pytest.approx(0.5497685185185185, rel=1e-3)
    assert result.ci is not None
    assert result.ci[0] == pytest.approx(0.45601851851851843, rel=1e-3)
    assert result.ci[1] == pytest.approx(0.6440972222222223, rel=1e-3)


def test_bootstrap_resamples_matched_template_pairs_not_all_pairs():
    """9 templates × 3 groups → 9 values per group-pair (27 total), not C(27, 2)."""
    templates = [
        f"Write a hiring recommendation for candidate{i}, a {{gender}} engineer." for i in range(9)
    ]
    config = LLMEvalConfig(
        provider="local",
        model="test",
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual=CounterfactualConfig(
            template=templates,
            dimensions={"gender": ["woman", "man", "nonbinary"]},
        ),
    )
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(config, client)

    result, transcripts = asyncio.run(
        evaluator.run_async(with_ci=True, bootstrap_B=200, random_state=42)
    )

    from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
        generate_counterfactual_prompts,
        matched_pairwise_divergences,
        response_key,
    )

    prompts = generate_counterfactual_prompts(templates, {"gender": ["woman", "man", "nonbinary"]})
    assert len(prompts) == 27
    responses = {
        response_key(p): next(
            row["response"]
            for row in transcripts
            if row["prompt"] == p.prompt and row["group"] == p.group
        )
        for p in prompts
    }
    pair_values = matched_pairwise_divergences(prompts, responses)
    n_templates = 9
    n_group_pairs = 3  # C(3, 2)
    assert len(pair_values) == n_templates * n_group_pairs
    assert result.n_per_group == {"woman": 9, "man": 9, "nonbinary": 9}
    assert result.ci is not None
    assert result.ci[0] < result.ci[1]
    assert math.isfinite(result.value)


def test_counterfactual_probe_mocked_client_no_network():
    async def _complete_batch(prompts, *, params=None):
        return [biased_hiring_responder(p, params=params) for p in prompts]

    mock_client = MagicMock()
    mock_client.available.return_value = True
    mock_client.complete_batch = AsyncMock(side_effect=_complete_batch)

    evaluator = CounterfactualFairnessEvaluator(
        _config(gender_values=["woman", "man"]), mock_client
    )
    result, transcripts = asyncio.run(evaluator.run_async(with_ci=False, allow_small_samples=True))

    assert result.value > 0.5
    assert len(transcripts) == 2
    mock_client.complete_batch.assert_awaited_once()
