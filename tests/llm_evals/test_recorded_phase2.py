"""Phase 2 recorded-cache replay. Refusal is live data (not illustrative); toxicity/BBQ remain BL-009."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.config import CounterfactualConfig
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_bbq_config,
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    humanitarian_divergence_config,
    populate_recorded_bbq_cache,
    populate_recorded_refusal_cache,
    populate_recorded_toxicity_cache,
    recorded_group_rates,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import caveat_for_cache_dir


def test_recorded_humanitarian_refusal_fixture_finite_at_default_threshold(
    assert_no_live_llm_calls,
):
    """Humanitarian refusal fixture: finite at min_group_size=5, no caveat. Not a disparity claim."""
    result = run_llm_eval(default_recorded_refusal_config(), with_ci=True, bootstrap_B=50)
    metric = result.metrics["refusal_rate_disparity"]
    assert math.isfinite(metric.value)
    assert metric.n_per_group == {"woman": 5, "man": 5, "ambiguous": 5}
    assert metric.caveat is None
    assert caveat_for_cache_dir(default_recorded_refusal_config().cache_dir) is None


def test_humanitarian_divergence_shares_refusal_counterfactual_block():
    """Divergence and refusal configs must stay on the same prompts/params/cache."""
    refusal = default_recorded_refusal_config()
    divergence = humanitarian_divergence_config()
    assert divergence.evaluators == ["demographic_swap_divergence"]
    assert divergence.cache_dir == refusal.cache_dir
    assert divergence.params == refusal.params
    assert divergence.provider == refusal.provider
    assert divergence.model == refusal.model
    assert divergence.counterfactual is not None
    assert refusal.counterfactual is not None
    assert divergence.counterfactual.template == refusal.counterfactual.template
    assert divergence.counterfactual.dimensions == refusal.counterfactual.dimensions
    assert divergence.counterfactual.name_pools == refusal.counterfactual.name_pools
    assert divergence.params == {"temperature": 0.0, "max_tokens": 512}


def test_humanitarian_divergence_replays_at_default_threshold(assert_no_live_llm_calls):
    """Humanitarian cache under divergence: finite at n=5/group, no caveat, CI present."""
    result = run_llm_eval(humanitarian_divergence_config(), with_ci=True, bootstrap_B=200)
    metric = result.metrics["demographic_swap_divergence"]
    assert math.isfinite(metric.value)
    assert 0.0 < metric.value < 1.0
    assert metric.n_per_group == {"woman": 5, "man": 5, "ambiguous": 5}
    assert metric.caveat is None
    assert metric.ci is not None
    assert metric.ci[0] < metric.ci[1]
    assert len(result.transcripts["counterfactual"]) == 15


def test_humanitarian_divergence_below_threshold_returns_nan(assert_no_live_llm_calls):
    """n=1/group humanitarian slice still replays the cache and the guard returns nan."""
    cfg = humanitarian_divergence_config()
    assert cfg.counterfactual is not None
    first_template = (
        cfg.counterfactual.template[0]
        if isinstance(cfg.counterfactual.template, list)
        else cfg.counterfactual.template
    )
    small = replace(
        cfg,
        counterfactual=CounterfactualConfig(
            template=[first_template],
            dimensions=dict(cfg.counterfactual.dimensions),
            name_pools={
                dim: {group: names[:1] for group, names in pools.items()}
                for dim, pools in cfg.counterfactual.name_pools.items()
            },
        ),
    )
    result = run_llm_eval(small, with_ci=False)
    metric = result.metrics["demographic_swap_divergence"]
    assert math.isnan(metric.value)
    assert metric.ci is None
    assert metric.n_per_group == {}


def test_recorded_toxicity_cache_replays_without_error(assert_no_live_llm_calls):
    """Committed hiring-cache copy replays; does not assert group-level toxicity signal (BL-009)."""
    result = run_llm_eval(default_recorded_toxicity_config(), with_ci=True, bootstrap_B=50)
    metric = result.metrics["toxicity_sentiment_disparity"]
    assert math.isfinite(metric.value)
    assert metric.n_per_group == {"woman": 9, "man": 9, "nonbinary": 9}
    assert metric.caveat is not None
    assert "BL-009" in metric.caveat


def test_recorded_bbq_cache_replays_without_error(assert_no_live_llm_calls):
    """Committed BBQ cache replays; all-ambiguous subset, not a disparity evidence claim (BL-009)."""
    result = run_llm_eval(default_recorded_bbq_config(), with_ci=True, bootstrap_B=50)
    metric = result.metrics["stereotype_association_score"]
    assert math.isfinite(metric.value)
    assert metric.n_per_group["woman"] >= 5
    assert metric.n_per_group["man"] >= 5
    assert metric.caveat is not None
    assert "BL-009" in metric.caveat


def test_refusal_manifest_prompt_count():
    from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_group_rates import (
        RECORDED_REFUSAL_MANIFEST_PATH,
    )

    manifest = json.loads(RECORDED_REFUSAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["prompts"]) == 15
    assert manifest.get("illustrative") is not True


def test_populate_recorded_toxicity_cache_copies_expanded(tmp_path, monkeypatch):
    """File copy only — not a live provider recording."""
    monkeypatch.setattr(recorded_group_rates, "RECORDED_TOXICITY_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        recorded_group_rates, "RECORDED_TOXICITY_MANIFEST_PATH", tmp_path / "manifest.json"
    )
    manifest = populate_recorded_toxicity_cache()
    assert len(manifest["prompts"]) == 27


@pytest.mark.live_llm
def test_populate_recorded_refusal_cache_live():
    import asyncio

    manifest = asyncio.run(populate_recorded_refusal_cache())
    assert len(manifest["prompts"]) == 15
    assert manifest.get("illustrative") is not True


@pytest.mark.live_llm
def test_populate_recorded_bbq_cache_live():
    import asyncio

    manifest = asyncio.run(populate_recorded_bbq_cache())
    assert len(manifest["prompts"]) >= 12
