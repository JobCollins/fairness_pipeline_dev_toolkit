"""Unit tests for the four-state LLM-eval gate helper."""

from __future__ import annotations

import math

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    EXIT_FAIL,
    EXIT_ILLUSTRATIVE,
    EXIT_PASS,
    EXIT_UNDEFINED,
    EXIT_USAGE,
    GATE_FAIL,
    GATE_ILLUSTRATIVE,
    GATE_PASS,
    GATE_STATUS_TO_EXIT,
    GATE_UNDEFINED,
    evaluate_llm_eval_gate,
)
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


def _metric(name: str, value: float, caveat: str | None = None) -> MetricResult:
    return MetricResult(metric=name, value=value, caveat=caveat)


def test_gate_pass_when_no_threshold_and_no_caveat():
    status, passed = evaluate_llm_eval_gate({"demographic_swap_divergence": _metric("cf", 0.19)})
    assert status == GATE_PASS
    assert passed is True


def test_gate_fail_on_threshold_miss_non_caveated():
    status, passed = evaluate_llm_eval_gate(
        {"demographic_swap_divergence": _metric("cf", 0.19)},
        threshold=0.01,
    )
    assert status == GATE_FAIL
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
    assert status == GATE_ILLUSTRATIVE
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
    assert status == GATE_ILLUSTRATIVE
    assert passed is None


def test_gate_undefined_on_nonfinite_value():
    status, passed = evaluate_llm_eval_gate(
        {"demographic_swap_divergence": _metric("cf", float("nan"))},
        threshold=0.10,
    )
    assert status == GATE_UNDEFINED
    assert passed is None
    assert math.isnan(float("nan"))


def test_gate_undefined_without_threshold():
    """Non-finite is never a silent pass, even when no threshold is set."""
    status, passed = evaluate_llm_eval_gate(
        {"demographic_swap_divergence": _metric("cf", float("nan"))}
    )
    assert status == GATE_UNDEFINED
    assert passed is None


def test_gate_illustrative_wins_over_undefined():
    """Caveated + nan (demo fixture below min_group_size): illustrative subsumes undefined."""
    status, passed = evaluate_llm_eval_gate(
        {
            "toxicity_sentiment_disparity": _metric(
                "toxicity_sentiment_disparity",
                float("nan"),
                caveat="Demo fixture (BL-009): not evidence.",
            )
        },
        threshold=0.05,
    )
    assert status == GATE_ILLUSTRATIVE
    assert passed is None


def test_gate_unknown_metric_raises_key_error():
    with pytest.raises(KeyError):
        evaluate_llm_eval_gate(
            {"demographic_swap_divergence": _metric("cf", 0.1)},
            metric="refusal_rate_disparity",
        )


def test_gate_status_cli_exit_mapping():
    assert GATE_STATUS_TO_EXIT[GATE_PASS] == EXIT_PASS == 0
    assert GATE_STATUS_TO_EXIT[GATE_FAIL] == EXIT_FAIL == 1
    assert EXIT_USAGE == 2
    assert GATE_STATUS_TO_EXIT[GATE_ILLUSTRATIVE] == EXIT_ILLUSTRATIVE == 3
    assert GATE_STATUS_TO_EXIT[GATE_UNDEFINED] == EXIT_UNDEFINED == 4
