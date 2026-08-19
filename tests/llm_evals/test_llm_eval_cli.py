"""CLI end-to-end tests for fairpipe llm-eval."""

from __future__ import annotations

import yaml

from fairness_pipeline_dev_toolkit.cli.main import main
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_counterfactual import (
    RECORDED_COUNTERFACTUAL_CACHE_DIR,
    RECORDED_COUNTERFACTUAL_DEFAULTS,
    RECORDED_COUNTERFACTUAL_DIMENSIONS,
    RECORDED_COUNTERFACTUAL_TEMPLATE,
    RECORDED_MODEL,
    RECORDED_PARAMS,
    RECORDED_PROVIDER,
)


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
