"""Unit tests for the three-state LLM-eval gate helper."""

from __future__ import annotations

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    EXIT_FAIL,
    EXIT_ILLUSTRATIVE,
    EXIT_PASS,
    EXIT_USAGE,
    GATE_STATUS_TO_EXIT,
    evaluate_llm_eval_gate,
)
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


def _metric(name: str, value: float, caveat: str | None = None) -> MetricResult:
    return MetricResult(metric=name, value=value, caveat=caveat)


def test_gate_pass_when_no_threshold_and_no_caveat():
    status, passed = evaluate_llm_eval_gate(
        {"counterfactual_fairness_divergence": _metric("cf", 0.19)}
    )
    assert status == "pass"
    assert passed is True


def test_gate_fail_on_threshold_miss_non_caveated():
    status, passed = evaluate_llm_eval_gate(
        {"counterfactual_fairness_divergence": _metric("cf", 0.19)},
        threshold=0.01,
    )
    assert status == "fail"
    assert passed is False


def test_gate_illustrative_even_if_number_would_pass_threshold():
    status, passed = evaluate_llm_eval_gate(
        {
            "refusal_rate_disparity": _metric(
                "refusal_rate_disparity",
                0.0,
                caveat="Demo fixture (BL-009): not evidence.",
            )
        },
        threshold=0.01,
    )
    assert status == "illustrative"
    assert passed is None


def test_gate_illustrative_even_if_number_would_fail_threshold():
    status, passed = evaluate_llm_eval_gate(
        {
            "toxicity_sentiment_disparity": _metric(
                "toxicity_sentiment_disparity",
                0.9,
                caveat="Demo fixture (BL-009): not evidence.",
            )
        },
        threshold=0.01,
    )
    assert status == "illustrative"
    assert passed is None


def test_gate_unknown_metric_raises_key_error():
    with pytest.raises(KeyError):
        evaluate_llm_eval_gate(
            {"counterfactual_fairness_divergence": _metric("cf", 0.1)},
            metric="refusal_rate_disparity",
        )


def test_gate_status_cli_exit_mapping():
    assert GATE_STATUS_TO_EXIT["pass"] == EXIT_PASS == 0
    assert GATE_STATUS_TO_EXIT["fail"] == EXIT_FAIL == 1
    assert EXIT_USAGE == 2
    assert GATE_STATUS_TO_EXIT["illustrative"] == EXIT_ILLUSTRATIVE == 3
