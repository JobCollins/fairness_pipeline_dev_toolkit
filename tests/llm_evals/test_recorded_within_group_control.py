"""Within-group control fixture: lexical baseline for counterfactual divergence (BL-012)."""

from __future__ import annotations

import itertools
from statistics import mean

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    load_recorded_within_group_control_manifest,
    load_within_group_control_records,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_within_group_control import (
    RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR,
    WITHIN_GROUP_CONTROL_NAMES,
    WITHIN_GROUP_CONTROL_PARAMS,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    extract_response_features,
    pairwise_divergence,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import caveat_for_cache_dir


def test_within_group_control_manifest_is_real_data():
    manifest = load_recorded_within_group_control_manifest()
    assert manifest["provider"] == "anthropic"
    assert manifest["model"] == "claude-haiku-4-5"
    assert manifest["params"] == WITHIN_GROUP_CONTROL_PARAMS
    assert manifest.get("illustrative") is not True
    assert len(manifest["prompts"]) == 9
    assert caveat_for_cache_dir(str(RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR)) is None


def test_within_group_control_cache_files_exist():
    manifest = load_recorded_within_group_control_manifest()
    for entry in manifest["prompts"]:
        path = RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR / f"{entry['cache_key']}.txt"
        assert path.is_file(), path
        assert path.read_text(encoding="utf-8").strip()


def _components(left: str, right: str) -> float:
    fa = extract_response_features(left, reference=right)
    fb = extract_response_features(right, reference=left)
    return pairwise_divergence(fa, fb)


def test_within_group_control_baseline_matches_recording():
    """Pinned evidence: within-group ≈ cross-group ≈ 0.19; 0 is not the baseline."""
    records = load_within_group_control_records()
    assert len(records) == 9
    assert {r["group"] for r in records} == set(WITHIN_GROUP_CONTROL_NAMES)
    within = []
    cross = []
    for left, right in itertools.combinations(records, 2):
        value = _components(left["response"], right["response"])
        if left["group"] == right["group"]:
            within.append(value)
        else:
            cross.append(value)
    assert len(within) == 9
    assert len(cross) == 27
    assert mean(within) == pytest.approx(0.190, abs=5e-3)
    assert mean(cross) == pytest.approx(0.187, abs=5e-3)
    # Control finding: within is not clearly below cross.
    assert mean(within) > mean(cross) - 0.02
