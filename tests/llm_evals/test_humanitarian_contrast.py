"""BL-012 Phase 3: humanitarian contrast recorded-cache replay."""

from __future__ import annotations

import asyncio
import math

import pytest

from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    humanitarian_contrast_config,
    populate_humanitarian_contrast_cache,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_group_rates import (
    RECORDED_HUMANITARIAN_CONTRAST_CACHE_DIR,
    RECORDED_HUMANITARIAN_CONTRAST_MANIFEST_PATH,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import caveat_for_cache_dir


def test_humanitarian_contrast_manifest_is_real_data():
    assert RECORDED_HUMANITARIAN_CONTRAST_MANIFEST_PATH.is_file()
    import json

    manifest = json.loads(RECORDED_HUMANITARIAN_CONTRAST_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest.get("illustrative") is not True
    assert manifest["evaluator"] == "counterfactual_fairness_contrast"
    assert manifest["counterfactual"]["control_dimension"] == "control"
    assert len(manifest["prompts"]) == 30
    assert caveat_for_cache_dir(str(RECORDED_HUMANITARIAN_CONTRAST_CACHE_DIR)) is None


def test_humanitarian_contrast_replays(assert_no_live_llm_calls):
    """Recorded contrast: finite signed value, both arms in n_per_group, CI, no caveat."""
    result = run_llm_eval(
        humanitarian_contrast_config(),
        with_ci=True,
        bootstrap_B=200,
        random_state=42,
    )
    metric = result.metrics["counterfactual_fairness_contrast"]
    assert math.isfinite(metric.value)
    # Near-zero / negative is the expected null — only a loose sanity bound.
    assert -0.5 < metric.value < 0.5
    assert metric.n_per_group == {
        "woman": 5,
        "man": 5,
        "ambiguous": 5,
        "Fatima": 5,
        "Amina": 5,
        "Leyla": 5,
    }
    assert metric.caveat is None
    assert metric.ci is not None
    assert metric.ci[0] < metric.ci[1]
    assert len(result.transcripts["counterfactual"]) == 30


@pytest.mark.live_llm
def test_populate_humanitarian_contrast_cache_live():
    """Re-record contrast fixture from live Anthropic API."""
    manifest = asyncio.run(populate_humanitarian_contrast_cache())
    assert manifest["provider"] == "anthropic"
    assert len(manifest["prompts"]) == 30
    assert manifest.get("illustrative") is not True
