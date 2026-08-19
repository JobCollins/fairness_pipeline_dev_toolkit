"""Tests for LLM eval dry-run estimation."""

from __future__ import annotations

import yaml

from fairness_pipeline_dev_toolkit.cli.main import main
from fairness_pipeline_dev_toolkit.llm_evals.dry_run import estimate_dry_run
from fairness_pipeline_dev_toolkit.llm_evals.runner import run_llm_eval


def test_estimate_dry_run_multiplies_by_template_count():
    estimate = estimate_dry_run(
        provider="anthropic",
        model="claude-haiku-4-5",
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual_dimensions={"gender": ["woman", "man", "nonbinary"]},
        n_templates=9,
    )
    assert estimate.request_count == 27
    assert estimate.breakdown["counterfactual:gender"] == 27


def test_estimate_dry_run_counts_phase2_evaluators():
    estimate = estimate_dry_run(
        provider="anthropic",
        model="claude-haiku-4-5",
        evaluators=[
            "refusal_rate_disparity",
            "toxicity_sentiment_disparity",
            "stereotype_association_score",
        ],
        counterfactual_dimensions={"gender": ["woman", "man", "nonbinary"]},
        n_templates=9,
        bbq_item_count=12,
    )
    assert estimate.request_count == 27 + 27 + 12
    assert estimate.breakdown["refusal_rate_disparity"] == 27
    assert estimate.breakdown["toxicity_sentiment_disparity"] == 27
    assert estimate.breakdown["stereotype_association_score"] == 12


def test_estimate_dry_run_counts_requests():
    estimate = estimate_dry_run(
        provider="openai",
        model="gpt-4o-mini",
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual_dimensions={"gender": ["woman", "man"]},
    )
    assert estimate.request_count == 2
    assert estimate.estimated_cost_usd > 0
    assert estimate.breakdown["counterfactual:gender"] == 2


def test_dry_run_makes_no_live_calls(tmp_path, capsys):
    cfg = tmp_path / "llm_eval.yml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "llm_eval": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "evaluators": ["counterfactual_fairness_divergence"],
                    "counterfactual": {
                        "template": "Recommend {name}, a {gender} engineer.",
                        "dimensions": {"gender": ["woman", "man"]},
                        "defaults": {"name": "Alex"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["llm-eval", "--config", str(cfg), "--dry-run"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Estimated requests" in captured.out
    assert "No live provider calls" in captured.out


def test_run_llm_eval_dry_run_returns_estimate():
    estimate_only = run_llm_eval(
        {
            "provider": "local",
            "model": "demo",
            "evaluators": ["counterfactual_fairness_divergence"],
            "counterfactual": {
                "template": "Recommend {name}, a {gender} engineer.",
                "dimensions": {"gender": ["woman", "man"]},
                "defaults": {"name": "Alex"},
            },
        },
        dry_run=True,
    )
    assert estimate_only.metrics == {}
    assert estimate_only.dry_run is not None
    assert estimate_only.dry_run.request_count == 2
