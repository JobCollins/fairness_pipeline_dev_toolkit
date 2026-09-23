"""assert_llm_fairness shares evaluate_llm_eval_gate with CLI / REST (BL-017)."""

from __future__ import annotations

import math

import pytest

from fairness_pipeline_dev_toolkit.integration.pytest_plugin import assert_llm_fairness
from fairness_pipeline_dev_toolkit.llm_evals.gating import (
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


def test_assert_llm_fairness_passes_within_threshold():
    assert_llm_fairness(0.05, 0.10)
    assert_llm_fairness(
        _metric("refusal_rate_disparity", 0.05),
        0.10,
    )


def test_assert_llm_fairness_raises_on_magnitude_violation():
    with pytest.raises(AssertionError, match="CLI exit 1|threshold exceeded"):
        assert_llm_fairness(0.15, 0.10)
    # Signed contrast: magnitude-based (matches CLI), not signed <= .
    with pytest.raises(AssertionError, match="magnitude-based|CLI exit 1"):
        assert_llm_fairness(
            _metric("demographic_swap_contrast", -0.05629),
            0.05,
        )


def test_assert_llm_fairness_caveated_raises_even_if_value_would_pass():
    with pytest.raises(AssertionError, match="illustrative|CLI exit 3"):
        assert_llm_fairness(
            _metric(
                "toxicity_sentiment_disparity",
                0.0,
                caveat="Demo fixture (BL-009): not evidence.",
            ),
            0.05,
        )


def test_assert_llm_fairness_nan_raises_as_undefined():
    """Non-finite → undefined (CLI exit 4); restores allow_nan=False raise-on-NaN."""
    with pytest.raises(AssertionError, match="undefined|CLI exit 4|min_group_size"):
        assert_llm_fairness(float("nan"), 0.10)
    with pytest.raises(AssertionError, match="undefined|CLI exit 4|min_group_size"):
        assert_llm_fairness(
            _metric("demographic_swap_divergence", float("nan")),
            0.10,
        )
    status, passed = evaluate_llm_eval_gate(
        {"m": _metric("m", float("nan"))},
        threshold=0.10,
    )
    assert status == GATE_UNDEFINED
    assert passed is None
    assert math.isnan(float("nan"))


def test_assert_llm_fairness_allow_nan_skips_undefined_raise():
    """allow_nan=True is a plugin-only opt-in to tolerate insufficient evidence."""
    assert_llm_fairness(float("nan"), 0.10, allow_nan=True)
    assert_llm_fairness(
        _metric("demographic_swap_divergence", float("nan")),
        0.10,
        allow_nan=True,
    )


def test_assert_llm_fairness_caveated_nan_raises_illustrative():
    """Precedence: illustrative wins over undefined when both apply."""
    with pytest.raises(AssertionError, match="illustrative|CLI exit 3"):
        assert_llm_fairness(
            _metric(
                "toxicity_sentiment_disparity",
                float("nan"),
                caveat="Demo fixture (BL-009): not evidence.",
            ),
            0.05,
        )


@pytest.mark.parametrize(
    "result,threshold,expected_status",
    [
        (_metric("toxicity_sentiment_disparity", 0.0, caveat="demo"), 0.05, GATE_ILLUSTRATIVE),
        (_metric("demographic_swap_contrast", -0.05629), 0.05, GATE_FAIL),
        (_metric("refusal_rate_disparity", 0.01), 0.05, GATE_PASS),
        (_metric("demographic_swap_divergence", float("nan")), 0.05, GATE_UNDEFINED),
        (
            _metric(
                "toxicity_sentiment_disparity",
                float("nan"),
                caveat="demo",
            ),
            0.05,
            GATE_ILLUSTRATIVE,
        ),
    ],
)
def test_assert_llm_fairness_agrees_with_evaluate_llm_eval_gate(
    result: MetricResult, threshold: float, expected_status: str
):
    """Cross-interface guard: plugin raise/pass matches shared gate status."""
    status, _passed = evaluate_llm_eval_gate(
        {result.metric: result},
        threshold=threshold,
        metric=result.metric,
    )
    assert status == expected_status
    assert GATE_STATUS_TO_EXIT[status] in (0, 1, 3, 4)

    if expected_status == GATE_PASS:
        assert_llm_fairness(result, threshold)
    else:
        with pytest.raises(AssertionError):
            assert_llm_fairness(result, threshold)
