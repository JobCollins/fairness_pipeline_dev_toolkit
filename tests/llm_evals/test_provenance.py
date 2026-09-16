"""MetricResult.caveat provenance from fixture manifest ``illustrative`` flag (BL-009)."""

from __future__ import annotations

import asyncio
import json
import math

from fairness_pipeline_dev_toolkit.integration.mlflow_logger import log_llm_eval_results
from fairness_pipeline_dev_toolkit.integration.reporting import to_markdown_report
from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.client import LocalLLMClient
from fairness_pipeline_dev_toolkit.llm_evals.config import (
    CounterfactualConfig,
    LLMEvalConfig,
)
from fairness_pipeline_dev_toolkit.llm_evals.demo import biased_hiring_responder
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.counterfactual_fairness import (
    CounterfactualFairnessEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR,
    default_recorded_bbq_config,
    default_recorded_counterfactual_config,
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    expanded_recorded_counterfactual_config,
    humanitarian_divergence_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import (
    CAVEAT_RECORDED_BBQ,
    CAVEAT_RECORDED_TOXICITY,
    DEFAULT_ILLUSTRATIVE_CAVEAT,
    caveat_for_cache_dir,
)
from fairness_pipeline_dev_toolkit.llm_evals.runner import results_to_markdown


def test_shipped_illustrative_fixtures_produce_caveat():
    assert caveat_for_cache_dir(default_recorded_toxicity_config().cache_dir) == (
        CAVEAT_RECORDED_TOXICITY
    )
    assert caveat_for_cache_dir(default_recorded_bbq_config().cache_dir) == CAVEAT_RECORDED_BBQ


def test_recorded_refusal_fixture_has_no_illustrative_caveat():
    """Humanitarian refusal fixture has no illustrative caveat (hiring-copy replaced)."""
    assert caveat_for_cache_dir(default_recorded_refusal_config().cache_dir) is None


def test_other_cache_paths_and_user_configs_do_not_produce_caveat(tmp_path):
    assert caveat_for_cache_dir(None) is None
    assert caveat_for_cache_dir(str(tmp_path / "cache")) is None
    lookalike = tmp_path / "recorded_refusal" / "cache"
    lookalike.mkdir(parents=True)
    assert caveat_for_cache_dir(str(lookalike)) is None
    assert caveat_for_cache_dir(expanded_recorded_counterfactual_config().cache_dir) is None
    assert caveat_for_cache_dir(default_recorded_counterfactual_config().cache_dir) is None
    assert caveat_for_cache_dir(default_recorded_refusal_config().cache_dir) is None
    assert caveat_for_cache_dir(humanitarian_divergence_config().cache_dir) is None
    assert caveat_for_cache_dir(str(RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR)) is None


def test_illustrative_false_on_same_layout_clears_caveat(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"illustrative": False, "caveat": "should be ignored"}),
        encoding="utf-8",
    )
    assert caveat_for_cache_dir(str(cache)) is None


def test_illustrative_true_on_arbitrary_path_sets_caveat(tmp_path):
    """Trigger is the manifest flag, not a hardcoded fixture path."""
    cache = tmp_path / "my_cache"
    cache.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"illustrative": True}),
        encoding="utf-8",
    )
    assert caveat_for_cache_dir(str(cache)) == DEFAULT_ILLUSTRATIVE_CAVEAT


def test_markdown_surface_renders_caveat_not_bare_number(assert_no_live_llm_calls):
    result = run_llm_eval(default_recorded_toxicity_config(), with_ci=False)
    report = results_to_markdown(result)
    assert result.metrics["toxicity_sentiment_disparity"].caveat == CAVEAT_RECORDED_TOXICITY
    assert "0.000000*" in report
    assert "## Caveats" in report
    assert CAVEAT_RECORDED_TOXICITY in report


def test_markdown_omits_caveat_section_without_flag():
    from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

    report = to_markdown_report(
        {"demographic_parity_difference": MetricResult(metric="dpd", value=0.12)}
    )
    assert "0.120000" in report
    assert "*" not in report.split("0.120000")[1][:2]
    assert "## Caveats" not in report


def test_mlflow_surface_renders_caveat_tag(mlflow_sqlite_tracking, assert_no_live_llm_calls):
    import mlflow

    result = run_llm_eval(default_recorded_toxicity_config(), with_ci=False)
    mlflow.set_experiment("llm_eval_caveat")
    with mlflow.start_run() as run:
        ok = log_llm_eval_results(result.metrics)
        run_id = run.info.run_id
    assert ok is True
    tags = mlflow.tracking.MlflowClient().get_run(run_id).data.tags
    assert tags["llm_eval.toxicity_sentiment_disparity.caveat"] == CAVEAT_RECORDED_TOXICITY


def test_mlflow_omits_caveat_tag_when_field_empty(mlflow_sqlite_tracking):
    import mlflow

    from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

    mlflow.set_experiment("llm_eval_no_caveat")
    with mlflow.start_run() as run:
        log_llm_eval_results(
            {"refusal_rate_disparity": MetricResult(metric="refusal_rate_disparity", value=0.12)}
        )
        run_id = run.info.run_id
    tags = mlflow.tracking.MlflowClient().get_run(run_id).data.tags
    assert "llm_eval.refusal_rate_disparity.caveat" not in tags


def _local_divergence_config(cache_dir: str) -> LLMEvalConfig:
    return LLMEvalConfig(
        provider="local",
        model="test",
        evaluators=["counterfactual_fairness_divergence"],
        counterfactual=CounterfactualConfig(
            template="Write a hiring recommendation for {name}, a {gender} engineer.",
            dimensions={"gender": ["woman", "man"]},
            defaults={"name": "Alex"},
        ),
        cache_dir=cache_dir,
        allow_small_samples=True,
    )


def test_divergence_evaluator_sets_caveat_when_manifest_illustrative(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"illustrative": True, "caveat": "Demo fixture (BL-009): divergence flag."}),
        encoding="utf-8",
    )
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_local_divergence_config(str(cache)), client)
    metric, _ = asyncio.run(evaluator.run_async(with_ci=False, allow_small_samples=True))
    assert metric.caveat == "Demo fixture (BL-009): divergence flag."


def test_divergence_evaluator_nan_path_sets_caveat_when_illustrative(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"illustrative": True, "caveat": "Demo fixture (BL-009): divergence flag."}),
        encoding="utf-8",
    )
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_local_divergence_config(str(cache)), client)
    metric, _ = asyncio.run(evaluator.run_async(with_ci=False, allow_small_samples=False))
    assert math.isnan(metric.value)
    assert metric.caveat == "Demo fixture (BL-009): divergence flag."


def test_divergence_evaluator_user_cache_has_no_caveat(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    client = LocalLLMClient("test", responder=biased_hiring_responder)
    evaluator = CounterfactualFairnessEvaluator(_local_divergence_config(str(cache)), client)
    metric, _ = asyncio.run(evaluator.run_async(with_ci=False, allow_small_samples=True))
    assert metric.caveat is None
