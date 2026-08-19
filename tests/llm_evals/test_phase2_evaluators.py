"""Phase 2 evaluator unit tests (mocked / local responders)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.bbq import (
    BBQ_LICENSE,
    BBQ_PINNED_COMMIT,
    BBQ_UPSTREAM_REPO,
    load_bbq_items,
    stereotypical_answer_index,
)
from fairness_pipeline_dev_toolkit.llm_evals.client import LocalLLMClient
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
)
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.refusal import (
    RefusalRateEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.stereotype import (
    StereotypeAssociationEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.toxicity import (
    ToxicitySentimentEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.scoring import (
    rate_disparity,
    refusal_score,
)


def _cf_config() -> LLMEvalConfig:
    return LLMEvalConfig(
        provider="local",
        model="test",
        evaluators=["refusal_rate_disparity"],
        counterfactual=CounterfactualConfig(
            template=["Prompt {gender}."] * 6,
            dimensions={"gender": ["woman", "man"]},
            defaults={},
        ),
    )


def test_refusal_score_known_phrases():
    assert refusal_score("I cannot assist with that request.") == 1.0
    assert refusal_score("Alex is a strong hire.") == 0.0


def test_refusal_rate_disparity_known_output():
    def responder(prompt: str, **kwargs) -> str:
        return "I cannot help with that." if "woman" in prompt else "Yes, hire them."

    client = LocalLLMClient("test", responder=responder)
    metric, rows = asyncio.run(RefusalRateEvaluator(_cf_config(), client).run_async(with_ci=False))
    assert metric.value == pytest.approx(1.0)
    assert metric.n_per_group == {"woman": 6, "man": 6}
    assert metric.caveat is None
    assert len(rows) == 12


def test_toxicity_rate_disparity_known_output():
    def responder(prompt: str, **kwargs) -> str:
        return "unqualified weak" if "woman" in prompt else "excellent candidate"

    client = LocalLLMClient("test", responder=responder)
    metric, _ = asyncio.run(
        ToxicitySentimentEvaluator(_cf_config(), client).run_async(with_ci=False)
    )
    assert metric.value > 0.0
    assert metric.n_per_group == {"woman": 6, "man": 6}


def test_rate_disparity_is_max_min_of_group_means_not_matched_pairs():
    """Refusal/toxicity/BBQ rates use unpaired group means (DPD-style), not template pairs."""
    scores = {"a": [1.0, 1.0, 0.0], "b": [0.0, 0.0, 0.0]}
    assert rate_disparity(scores) == pytest.approx(2.0 / 3.0)


def test_stereotype_known_output_selects_stereotyped_index():
    items = []
    for i in range(6):
        for group in ("woman", "man"):
            items.append(
                {
                    "category": "Gender_identity",
                    "context": f"Item {i}",
                    "question": "Who?",
                    "ans0": "A woman",
                    "ans1": "A man",
                    "ans2": "Unknown",
                    "label": 2,
                    "group": group,
                    "additional_metadata": {"stereotyped_groups": ["man"]},
                }
            )
    assert stereotypical_answer_index(items[0]) == 1

    def responder(prompt: str, **kwargs) -> str:
        return "1" if "Item 0" in prompt or "Item 1" in prompt else "2"

    client = LocalLLMClient("test", responder=responder)
    evaluator = StereotypeAssociationEvaluator(
        LLMEvalConfig(provider="local", model="test", evaluators=["stereotype_association_score"]),
        client,
        items=items,
    )
    metric, _ = asyncio.run(evaluator.run_async(with_ci=False))
    assert metric.value == pytest.approx(0.0)  # same stereotyped-hit rate in both groups
    assert metric.n_per_group == {"woman": 6, "man": 6}


def test_unparseable_stereotype_response_is_miss_not_unknown():
    """Unparseable output scores 0 (not a silent default to gold-unknown / label 2)."""
    from fairness_pipeline_dev_toolkit.llm_evals.evaluators.stereotype import (
        _parse_choice,
    )

    items = [
        {
            "category": "Gender_identity",
            "context": "Ambiguous.",
            "question": "Who?",
            "ans0": "A woman",
            "ans1": "A man",
            "ans2": "Unknown",
            "label": 2,
            "group": group,
            "additional_metadata": {"stereotyped_groups": ["man"]},
        }
        for group in (["woman"] * 5 + ["man"] * 5)
    ]

    client = LocalLLMClient(
        "test", responder=lambda prompt, **kwargs: "I decline to pick a number."
    )
    evaluator = StereotypeAssociationEvaluator(
        LLMEvalConfig(provider="local", model="test", evaluators=["stereotype_association_score"]),
        client,
        items=items,
    )
    metric, rows = asyncio.run(evaluator.run_async(with_ci=False))
    assert all(_parse_choice(r["response"]) is None for r in rows)
    assert metric.value == pytest.approx(0.0)


def test_local_bbq_fixture_is_all_ambiguous_gold_unknown():
    items = load_bbq_items()
    assert len(items) == 12
    assert all(item.get("label") == 2 for item in items)


def test_bbq_loader_uses_local_fixture_by_default():
    items = load_bbq_items()
    groups = {}
    for item in items:
        groups[item["group"]] = groups.get(item["group"], 0) + 1
    assert groups["woman"] >= 5
    assert groups["man"] >= 5
    assert all("context" in item and "question" in item for item in items)


@pytest.mark.live_bbq
def test_bbq_fetch_upstream_from_pinned_commit():
    items = load_bbq_items(fetch_upstream=True, max_items=8)
    assert len(items) == 8
    assert "question" in items[0]


def test_bbq_attribution_files_exist():
    root = Path(__file__).resolve().parents[2]
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    attribution = (root / "ATTRIBUTION.md").read_text(encoding="utf-8")
    for blob in (notice, attribution):
        assert "BBQ" in blob
        assert BBQ_LICENSE in blob
        assert BBQ_UPSTREAM_REPO in blob
        assert BBQ_PINNED_COMMIT in blob or "nyu-mll/BBQ" in blob
