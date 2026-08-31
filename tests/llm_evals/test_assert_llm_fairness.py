"""assert_llm_fairness mirrors assert_fairness."""

from __future__ import annotations

import pytest

from fairness_pipeline_dev_toolkit.integration.pytest_plugin import assert_llm_fairness
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


def test_assert_llm_fairness_passes_within_threshold():
    assert_llm_fairness(0.05, 0.10, comparator="<=")
    assert_llm_fairness(
        MetricResult(metric="refusal_rate_disparity", value=0.05, ci=None, effect_size=0.05),
        0.10,
    )


def test_assert_llm_fairness_raises_on_violation():
    with pytest.raises(AssertionError, match="Fairness threshold exceeded"):
        assert_llm_fairness(0.15, 0.10, comparator="<=")


def test_assert_llm_fairness_nan_policy():
    with pytest.raises(AssertionError, match="NaN"):
        assert_llm_fairness(float("nan"), 0.10, allow_nan=False)
    assert_llm_fairness(float("nan"), 0.10, allow_nan=True)
