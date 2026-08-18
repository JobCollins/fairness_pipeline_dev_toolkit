"""Tests for committed live-recorded counterfactual cache replay."""

from __future__ import annotations

import math

import pytest

from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_counterfactual_config,
    load_recorded_manifest,
    populate_recorded_counterfactual_cache,
)
from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE

# Canonical divergence from Anthropic claude-haiku-4-5 (temperature=0) recorded fixture.
RECORDED_DIVERGENCE = 0.18691293187798266


def test_recorded_manifest_exists():
    manifest = load_recorded_manifest()
    assert manifest["provider"] == "anthropic"
    assert manifest["model"] == "claude-haiku-4-5"
    assert len(manifest["prompts"]) == 3


def test_recorded_fixture_blocked_at_default_min_group_size():
    config = default_recorded_counterfactual_config()
    result = run_llm_eval(config, with_ci=False)

    metric = result.metrics["counterfactual_fairness_divergence"]
    assert math.isnan(metric.value)
    assert metric.n_per_group == {}


def test_recorded_cache_replay_allow_small_samples():
    """Replay committed cache with illustrative override — one prompt per group."""
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


@pytest.mark.live_llm
def test_populate_recorded_counterfactual_cache_live():
    """Regenerate fixture cache from live Anthropic API (run once, then commit cache files)."""
    import asyncio

    manifest = asyncio.run(populate_recorded_counterfactual_cache())
    assert manifest["provider"] == "anthropic"
    assert len(manifest["prompts"]) == 3
