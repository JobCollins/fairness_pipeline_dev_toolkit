"""Phase 2 recorded-cache replay. Refusal is live data (not illustrative); toxicity/BBQ remain BL-009."""

from __future__ import annotations

import json
import math

import pytest

from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_bbq_config,
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
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
