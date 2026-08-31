"""CLI end-to-end tests for fairpipe llm-eval."""

from __future__ import annotations

import time

import pytest
import yaml

from fairness_pipeline_dev_toolkit.cli.main import main
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_bbq_config,
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    expanded_recorded_counterfactual_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_counterfactual import (
    RECORDED_COUNTERFACTUAL_CACHE_DIR,
    RECORDED_COUNTERFACTUAL_DEFAULTS,
    RECORDED_COUNTERFACTUAL_DIMENSIONS,
    RECORDED_COUNTERFACTUAL_TEMPLATE,
    RECORDED_MODEL,
    RECORDED_PARAMS,
    RECORDED_PROVIDER,
)
from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    EXIT_FAIL,
    EXIT_ILLUSTRATIVE,
    EXIT_PASS,
    EXIT_USAGE,
)

from .conftest import write_llm_eval_yaml


def _write_config(tmp_path):
    cfg = tmp_path / "llm_eval.yml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "llm_eval": {
                    "provider": RECORDED_PROVIDER,
                    "model": RECORDED_MODEL,
                    "evaluators": ["counterfactual_fairness_divergence"],
                    "counterfactual": {
                        "template": RECORDED_COUNTERFACTUAL_TEMPLATE,
                        "dimensions": RECORDED_COUNTERFACTUAL_DIMENSIONS,
                        "defaults": RECORDED_COUNTERFACTUAL_DEFAULTS,
                    },
                    "cache_dir": str(RECORDED_COUNTERFACTUAL_CACHE_DIR),
                    "params": dict(RECORDED_PARAMS),
                }
            }
        ),
        encoding="utf-8",
    )
    return cfg


def test_llm_eval_cli_writes_markdown_report(tmp_path, capsys, assert_no_live_llm_calls):
    cfg = _write_config(tmp_path)
    report = tmp_path / "llm_report.md"

    exit_code = main(
        [
            "llm-eval",
            "--config",
            str(cfg),
            "--report-md",
            str(report),
            "--allow-small-samples",
        ]
    )

    assert exit_code == 0
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "# LLM Fairness Evaluation Report" in content
    assert "counterfactual_fairness_divergence" in content
    assert "0." in content


def test_llm_eval_cli_dry_run_prints_estimate(tmp_path, capsys):
    cfg = _write_config(tmp_path)
    exit_code = main(["llm-eval", "--config", str(cfg), "--dry-run"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Estimated requests" in captured.out
    assert "Estimated cost" in captured.out


def test_llm_eval_cli_threshold_pass_exit_zero(tmp_path, capsys, assert_no_live_llm_calls):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = main(
        [
            "llm-eval",
            "--config",
            str(cfg),
            "--metric",
            "counterfactual_fairness_divergence",
            "--threshold",
            "0.50",
        ]
    )
    assert exit_code == EXIT_PASS


def test_llm_eval_cli_threshold_fail_exit_one(tmp_path, capsys, assert_no_live_llm_calls):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = main(
        [
            "llm-eval",
            "--config",
            str(cfg),
            "--metric",
            "counterfactual_fairness_divergence",
            "--threshold",
            "0.01",
        ]
    )
    assert exit_code == EXIT_FAIL


@pytest.mark.parametrize(
    "factory,metric",
    [
        (default_recorded_refusal_config, "refusal_rate_disparity"),
        (default_recorded_toxicity_config, "toxicity_sentiment_disparity"),
        (default_recorded_bbq_config, "stereotype_association_score"),
    ],
    ids=["refusal", "toxicity", "bbq"],
)
def test_llm_eval_cli_recorded_illustrative_exit_three_even_if_threshold_would_pass(
    tmp_path, capsys, assert_no_live_llm_calls, factory, metric
):
    """BL-009 demo fixtures: caveat wins over a threshold the number would pass."""
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", factory())
    exit_code = main(
        [
            "llm-eval",
            "--config",
            str(cfg),
            "--metric",
            metric,
            "--threshold",
            "0.01",
        ]
    )
    assert exit_code == EXIT_ILLUSTRATIVE


def test_llm_eval_cli_threshold_without_metric_exit_two(tmp_path, capsys):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = main(
        [
            "llm-eval",
            "--config",
            str(cfg),
            "--threshold",
            "0.05",
        ]
    )
    assert exit_code == EXIT_USAGE
    captured = capsys.readouterr()
    assert "--metric is required" in captured.err


def test_llm_eval_cli_unknown_metric_exit_two(tmp_path, capsys):
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", expanded_recorded_counterfactual_config())
    exit_code = main(
        [
            "llm-eval",
            "--config",
            str(cfg),
            "--metric",
            "not_a_real_metric",
            "--threshold",
            "0.05",
        ]
    )
    assert exit_code == EXIT_USAGE
    captured = capsys.readouterr()
    assert "not_a_real_metric" in captured.err


def test_llm_eval_cli_cache_miss_nonzero_instant(tmp_path, capsys, assert_no_live_llm_calls):
    config = LLMEvalConfig(
        provider=RECORDED_PROVIDER,
        model=RECORDED_MODEL,
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual=CounterfactualConfig(
            template=RECORDED_COUNTERFACTUAL_TEMPLATE,
            dimensions={"gender": ["woman", "man"]},
            defaults=dict(RECORDED_COUNTERFACTUAL_DEFAULTS),
        ),
        cache_dir=str(tmp_path / "empty_cache"),
        params=dict(RECORDED_PARAMS),
        allow_small_samples=True,
    )
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", config)
    t0 = time.perf_counter()
    exit_code = main(["llm-eval", "--config", str(cfg), "--allow-small-samples"])
    elapsed = time.perf_counter() - t0
    assert exit_code != 0
    assert elapsed < 2.0
    captured = capsys.readouterr()
    assert "Cache miss" in captured.err or "CacheMiss" in captured.err


def test_llm_eval_cli_missing_cache_dir_nonzero_instant(tmp_path, capsys):
    config = LLMEvalConfig(
        provider=RECORDED_PROVIDER,
        model=RECORDED_MODEL,
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual=CounterfactualConfig(
            template=RECORDED_COUNTERFACTUAL_TEMPLATE,
            dimensions={"gender": ["woman", "man"]},
            defaults=dict(RECORDED_COUNTERFACTUAL_DEFAULTS),
        ),
        cache_dir=None,
        params={"temperature": 0.0, "max_tokens": 16},
        allow_small_samples=True,
    )
    cfg = write_llm_eval_yaml(tmp_path / "llm_eval.yml", config)
    t0 = time.perf_counter()
    exit_code = main(["llm-eval", "--config", str(cfg), "--allow-small-samples"])
    elapsed = time.perf_counter() - t0
    assert exit_code != 0
    assert elapsed < 2.0
    captured = capsys.readouterr()
    assert "LiveLLMCallForbidden" in captured.err or "forbidden" in captured.err.lower()
