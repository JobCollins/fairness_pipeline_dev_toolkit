"""Tests for committed live-recorded counterfactual cache replay."""

from __future__ import annotations

import math

import pytest

from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_counterfactual_config,
    expanded_recorded_counterfactual_config,
    load_expanded_recorded_manifest,
    load_recorded_manifest,
    populate_expanded_recorded_counterfactual_cache,
    populate_recorded_counterfactual_cache,
)
from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    generate_counterfactual_prompts,
    matched_pairwise_divergences,
    response_key,
)

# Canonical n=1 divergence from Anthropic claude-haiku-4-5 (temperature=0) recorded fixture.
RECORDED_DIVERGENCE = 0.18691293187798266


def test_recorded_manifest_exists():
    manifest = load_recorded_manifest()
    assert manifest["provider"] == "anthropic"
    assert manifest["model"] == "claude-haiku-4-5"
    assert len(manifest["prompts"]) == 3


def test_recorded_fixture_blocked_at_default_min_group_size(assert_no_live_llm_calls):
    config = default_recorded_counterfactual_config()
    result = run_llm_eval(config, with_ci=False)

    metric = result.metrics["counterfactual_fairness_divergence"]
    assert math.isnan(metric.value)
    assert metric.n_per_group == {}


def test_recorded_cache_replay_allow_small_samples(assert_no_live_llm_calls):
    """Replay n=1 committed cache with illustrative override — one prompt per group."""
    config = default_recorded_counterfactual_config()
    result = run_llm_eval(
        config,
        allow_small_samples=True,
        with_ci=False,
    )
    metric = result.metrics["counterfactual_fairness_divergence"]
    assert metric.value == pytest.approx(RECORDED_DIVERGENCE, rel=1e-6)
    assert metric.n_per_group == {"woman": 1, "man": 1, "nonbinary": 1}
    assert len(result.transcripts["counterfactual"]) == 3
    assert DEFAULT_LLM_MIN_GROUP_SIZE == 5


def test_expanded_recorded_fixture_finite_at_default_threshold(assert_no_live_llm_calls):
    """BL-007: expanded fixture clears min_group_size=5 without allow_small_samples."""
    config = expanded_recorded_counterfactual_config()
    result = run_llm_eval(config, with_ci=True, bootstrap_B=200, random_state=42)
    metric = result.metrics["counterfactual_fairness_divergence"]

    assert math.isfinite(metric.value)
    assert metric.n_per_group == {"woman": 9, "man": 9, "nonbinary": 9}
    assert metric.ci is not None
    assert metric.ci[0] < metric.ci[1]
    assert len(result.transcripts["counterfactual"]) == 27

    prompts = generate_counterfactual_prompts(
        config.counterfactual.template,
        config.counterfactual.dimensions,
        config.counterfactual.defaults,
    )
    responses = {
        response_key(item): next(
            row["response"]
            for row in result.transcripts["counterfactual"]
            if row["prompt"] == item.prompt and row["group"] == item.group
        )
        for item in prompts
    }
    pair_values = matched_pairwise_divergences(prompts, responses)
    assert len(pair_values) == 27  # 9 templates × C(3, 2) group-pairs
    assert metric.caveat is None


def test_expanded_recorded_manifest_exists():
    manifest = load_expanded_recorded_manifest()
    assert manifest["provider"] == "anthropic"
    assert len(manifest["prompts"]) == 27


@pytest.mark.live_llm
def test_populate_recorded_counterfactual_cache_live():
    """Regenerate n=1 fixture cache from live Anthropic API."""
    import asyncio

    manifest = asyncio.run(populate_recorded_counterfactual_cache())
    assert manifest["provider"] == "anthropic"
    assert len(manifest["prompts"]) == 3


@pytest.mark.live_llm
def test_populate_expanded_recorded_counterfactual_cache_live():
    """Regenerate expanded fixture cache from live Anthropic API."""
    import asyncio

    manifest = asyncio.run(populate_expanded_recorded_counterfactual_cache())
    assert manifest["provider"] == "anthropic"
    assert len(manifest["prompts"]) == 27
