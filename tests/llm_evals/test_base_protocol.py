"""Tests for LLMEvalAdapter protocol contract."""

from __future__ import annotations

from fairness_pipeline_dev_toolkit.llm_evals.base import (
    LLMEvalAdapter,
    StubLLMEvalAdapter,
)


def test_stub_satisfies_llm_eval_adapter_protocol():
    adapter = StubLLMEvalAdapter()
    assert isinstance(adapter, LLMEvalAdapter)


def test_stub_available_returns_true():
    adapter = StubLLMEvalAdapter()
    assert adapter.available() is True


def test_stub_counterfactual_returns_metric_result():
    adapter = StubLLMEvalAdapter()
    result = adapter.counterfactual_fairness_divergence(min_group_size=10)

    assert result.metric == "counterfactual_fairness_divergence"
    assert isinstance(result.value, float)
    assert hasattr(result, "ci")
    assert hasattr(result, "effect_size")
    assert hasattr(result, "n_per_group")
    assert result.n_per_group == {"A": 10, "B": 10}
