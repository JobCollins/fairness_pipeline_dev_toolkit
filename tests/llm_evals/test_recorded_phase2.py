"""Phase 2 recorded-cache replay (pipeline smoke, not disparity evidence — see BL-009)."""

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


def test_recorded_refusal_cache_replays_without_error(assert_no_live_llm_calls):
    """Committed hiring-cache copy replays; does not assert group-level refusal signal (BL-009)."""
    result = run_llm_eval(default_recorded_refusal_config(), with_ci=True, bootstrap_B=50)
    metric = result.metrics["refusal_rate_disparity"]
    assert math.isfinite(metric.value)
    assert metric.n_per_group == {"woman": 9, "man": 9, "nonbinary": 9}
    assert metric.caveat is not None
    assert "BL-009" in metric.caveat


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
    assert len(manifest["prompts"]) == 27


def test_populate_recorded_refusal_cache_copies_expanded(tmp_path, monkeypatch):
    """File copy only — not a live provider recording."""
    monkeypatch.setattr(recorded_group_rates, "RECORDED_REFUSAL_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        recorded_group_rates, "RECORDED_REFUSAL_MANIFEST_PATH", tmp_path / "manifest.json"
    )
    manifest = populate_recorded_refusal_cache()
    assert len(manifest["prompts"]) == 27
    assert (tmp_path / "cache").exists()
    assert len(list((tmp_path / "cache").glob("*.txt"))) == 27


def test_populate_recorded_toxicity_cache_copies_expanded(tmp_path, monkeypatch):
    """File copy only — not a live provider recording."""
    monkeypatch.setattr(recorded_group_rates, "RECORDED_TOXICITY_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        recorded_group_rates, "RECORDED_TOXICITY_MANIFEST_PATH", tmp_path / "manifest.json"
    )
    manifest = populate_recorded_toxicity_cache()
    assert len(manifest["prompts"]) == 27


@pytest.mark.live_llm
def test_populate_recorded_bbq_cache_live():
    import asyncio

    manifest = asyncio.run(populate_recorded_bbq_cache())
    assert len(manifest["prompts"]) >= 12
