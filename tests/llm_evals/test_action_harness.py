"""Local Action harness: Action-shaped inputs → reserved CLI exit codes.

No real GitHub Actions run. ``SvrusIO/fairpipe-action`` is a separate repo (BL-010).
"""

from __future__ import annotations

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.action_harness import (
    run_llm_fairness_check,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_bbq_config,
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    expanded_recorded_counterfactual_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    EXIT_FAIL,
    EXIT_ILLUSTRATIVE,
    EXIT_PASS,
    EXIT_USAGE,
)

from .conftest import write_llm_eval_yaml


def test_action_harness_pass_exit_zero(tmp_path, capsys, assert_no_live_llm_calls):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = run_llm_fairness_check(
        {
            "config": str(cfg),
            "metric": "counterfactual_fairness_divergence",
            "threshold": "0.50",
            "fail-on-violation": "true",
        }
    )
    assert exit_code == EXIT_PASS


def test_action_harness_fail_exit_one(tmp_path, capsys, assert_no_live_llm_calls):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = run_llm_fairness_check(
        {
            "config": str(cfg),
            "metric": "counterfactual_fairness_divergence",
            "threshold": "0.01",
            "fail-on-violation": "true",
        }
    )
    assert exit_code == EXIT_FAIL


def test_action_harness_usage_threshold_without_metric_exit_two(
    tmp_path, capsys, assert_no_live_llm_calls
):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = run_llm_fairness_check(
        {
            "config": str(cfg),
            "threshold": "0.05",
            "fail-on-violation": "true",
        }
    )
    assert exit_code == EXIT_USAGE


@pytest.mark.parametrize(
    "factory,metric",
    [
        (default_recorded_refusal_config, "refusal_rate_disparity"),
        (default_recorded_toxicity_config, "toxicity_sentiment_disparity"),
        (default_recorded_bbq_config, "stereotype_association_score"),
    ],
    ids=["refusal", "toxicity", "bbq"],
)
def test_action_harness_illustrative_exit_three_even_if_threshold_would_pass(
    tmp_path, capsys, assert_no_live_llm_calls, factory, metric
):
    """Caveated metric stays exit 3 even when the number would pass the threshold."""
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", factory())
    exit_code = run_llm_fairness_check(
        {
            "config": str(cfg),
            "metric": metric,
            "threshold": "0.01",
            "fail-on-violation": "true",
        }
    )
    assert exit_code == EXIT_ILLUSTRATIVE
