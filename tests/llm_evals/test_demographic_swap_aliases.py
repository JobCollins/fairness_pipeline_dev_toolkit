"""Deprecated-name aliases for demographic_swap_* metrics (one-release window)."""

from __future__ import annotations

import warnings

import pytest

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError
from fairness_pipeline_dev_toolkit.llm_evals import (
    EVALUATOR_ALIASES,
    VALID_EVALUATORS,
    canonicalize_evaluator_name,
    canonicalize_evaluators,
    evaluate_llm_eval_gate,
    load_llm_eval_config,
    run_llm_eval,
)
from fairness_pipeline_dev_toolkit.llm_evals.base import (
    LLMEvalAdapter,
    StubLLMEvalAdapter,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    expanded_recorded_counterfactual_config,
    humanitarian_contrast_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.names import (
    DEMOGRAPHIC_SWAP_CONTRAST,
    DEMOGRAPHIC_SWAP_DIVERGENCE,
)


def test_valid_evaluators_lists_canonical_only():
    assert DEMOGRAPHIC_SWAP_DIVERGENCE in VALID_EVALUATORS
    assert DEMOGRAPHIC_SWAP_CONTRAST in VALID_EVALUATORS
    assert "counterfactual_fairness_divergence" not in VALID_EVALUATORS
    assert "counterfactual_fairness_contrast" not in VALID_EVALUATORS


def test_aliases_map_to_canonical():
    assert EVALUATOR_ALIASES["counterfactual_fairness_divergence"] == DEMOGRAPHIC_SWAP_DIVERGENCE
    assert EVALUATOR_ALIASES["counterfactual_fairness_contrast"] == DEMOGRAPHIC_SWAP_CONTRAST


def test_canonicalize_warns_futurewarning():
    with pytest.warns(FutureWarning, match="demographic_swap_divergence"):
        assert (
            canonicalize_evaluator_name("counterfactual_fairness_divergence")
            == DEMOGRAPHIC_SWAP_DIVERGENCE
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            canonicalize_evaluator_name(DEMOGRAPHIC_SWAP_DIVERGENCE) == DEMOGRAPHIC_SWAP_DIVERGENCE
        )


def test_alias_futurewarning_visible_under_default_filters():
    """FutureWarning must fire under default filters (unlike DeprecationWarning)."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        canonicalize_evaluator_name("counterfactual_fairness_divergence")
    matching = [w for w in recorded if issubclass(w.category, FutureWarning)]
    assert matching, "expected FutureWarning under default filters"
    assert "demographic_swap_divergence" in str(matching[0].message)


def test_four_entry_points_resolve_identically():
    """Config / CLI / REST / gate all share :func:`canonicalize_evaluator_name`."""
    old = "counterfactual_fairness_divergence"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        via_direct = canonicalize_evaluator_name(old)
        via_list = canonicalize_evaluators([old])[0]
        cfg = load_llm_eval_config(
            obj={
                "provider": "local",
                "model": "demo",
                "evaluators": [old],
                "counterfactual": {
                    "template": "Hello {gender}",
                    "dimensions": {"gender": ["woman", "man"]},
                },
            }
        )
        via_config = cfg.evaluators[0]
        via_gate = canonicalize_evaluator_name(old)
    assert via_direct == via_list == via_config == via_gate == DEMOGRAPHIC_SWAP_DIVERGENCE


def test_yaml_alias_stores_canonical_and_output_keys_are_new_only(assert_no_live_llm_calls):
    cfg = expanded_recorded_counterfactual_config()
    raw = {
        "provider": cfg.provider,
        "model": cfg.model,
        "evaluators": ["counterfactual_fairness_divergence"],
        "counterfactual": {
            "template": cfg.counterfactual.template,
            "dimensions": cfg.counterfactual.dimensions,
            "defaults": cfg.counterfactual.defaults,
        },
        "cache_dir": cfg.cache_dir,
        "params": cfg.params,
    }
    with pytest.warns(FutureWarning, match="demographic_swap_divergence"):
        loaded = load_llm_eval_config(obj=raw)
    assert loaded.evaluators == [DEMOGRAPHIC_SWAP_DIVERGENCE]
    result = run_llm_eval(loaded, with_ci=False)
    assert DEMOGRAPHIC_SWAP_DIVERGENCE in result.metrics
    assert "counterfactual_fairness_divergence" not in result.metrics


def test_gate_accepts_old_metric_name_against_new_keys():
    metrics = {DEMOGRAPHIC_SWAP_DIVERGENCE: {"value": 0.1, "caveat": None}}
    with pytest.warns(FutureWarning, match="demographic_swap_divergence"):
        status, passed = evaluate_llm_eval_gate(
            metrics,
            threshold=0.25,
            metric="counterfactual_fairness_divergence",
        )
    assert status == "pass"
    assert passed is True


def test_old_output_key_raises_keyerror(assert_no_live_llm_calls):
    result = run_llm_eval(expanded_recorded_counterfactual_config(), with_ci=False)
    with pytest.raises(KeyError):
        _ = result.metrics["counterfactual_fairness_divergence"]


def test_deprecated_class_alias_warns():
    from fairness_pipeline_dev_toolkit.llm_evals.evaluators.counterfactual_fairness import (
        CounterfactualFairnessEvaluator,
    )
    from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
        default_recorded_counterfactual_config,
    )
    from fairness_pipeline_dev_toolkit.llm_evals.runner import build_client

    cfg = default_recorded_counterfactual_config()
    client = build_client(cfg)
    with pytest.warns(FutureWarning, match="DemographicSwapEvaluator"):
        CounterfactualFairnessEvaluator(cfg, client)


def test_legacy_adapter_methods_exist_as_deprecated_wrappers():
    """Protocol + stub keep old method names as FutureWarning wrappers."""
    assert hasattr(LLMEvalAdapter, "counterfactual_fairness_divergence")
    assert hasattr(LLMEvalAdapter, "counterfactual_fairness_contrast")
    adapter = StubLLMEvalAdapter()
    assert isinstance(adapter, LLMEvalAdapter)
    with pytest.warns(FutureWarning, match="demographic_swap_divergence"):
        result = adapter.counterfactual_fairness_divergence(min_group_size=5)
    assert result.metric == DEMOGRAPHIC_SWAP_DIVERGENCE
    with pytest.warns(FutureWarning, match="demographic_swap_contrast"):
        result = adapter.counterfactual_fairness_contrast(min_group_size=5)
    assert result.metric == DEMOGRAPHIC_SWAP_CONTRAST


def test_cli_prints_stderr_for_old_metric_name(tmp_path, capsys, assert_no_live_llm_calls):
    from fairness_pipeline_dev_toolkit.cli.llm_eval_cmd import cmd_llm_eval
    from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
        expanded_recorded_counterfactual_config,
    )

    cfg = expanded_recorded_counterfactual_config()
    config_path = tmp_path / "llm.yml"
    # Write canonical evaluators so only --metric triggers the alias stderr line once
    # for the metric flag (config load will not add another alias line).
    import yaml

    config_path.write_text(
        yaml.safe_dump(
            {
                "provider": cfg.provider,
                "model": cfg.model,
                "evaluators": [DEMOGRAPHIC_SWAP_DIVERGENCE],
                "counterfactual": {
                    "template": cfg.counterfactual.template,
                    "dimensions": cfg.counterfactual.dimensions,
                    "defaults": cfg.counterfactual.defaults,
                },
                "cache_dir": cfg.cache_dir,
                "params": cfg.params,
            }
        ),
        encoding="utf-8",
    )

    class Args:
        config = str(config_path)
        metric = "counterfactual_fairness_divergence"
        threshold = 1.0
        dry_run = False
        min_group_size = 5
        allow_small_samples = False
        with_ci = False
        ci_level = 0.95
        bootstrap_B = 50
        random_state = 42
        report_md = None
        out = None
        transcripts_out = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        code = cmd_llm_eval(Args())
    assert code == 0
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "counterfactual_fairness_divergence" in err
    assert "demographic_swap_divergence" in err


def test_rest_surfaces_deprecations(assert_no_live_llm_calls):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from fairness_pipeline_dev_toolkit.api.app import create_app

    cfg = expanded_recorded_counterfactual_config()
    payload = {
        "provider": cfg.provider,
        "model": cfg.model,
        "evaluators": ["counterfactual_fairness_divergence"],
        "cache_dir": cfg.cache_dir,
        "params": dict(cfg.params),
        "with_ci": False,
        "min_group_size": 5,
        "counterfactual": {
            "template": cfg.counterfactual.template,
            "dimensions": cfg.counterfactual.dimensions,
            "defaults": cfg.counterfactual.defaults,
        },
        "metric": "counterfactual_fairness_divergence",
        "threshold": 1.0,
    }
    with TestClient(create_app()) as client:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert DEMOGRAPHIC_SWAP_DIVERGENCE in body["metrics"]
    assert "counterfactual_fairness_divergence" not in body["metrics"]
    assert body.get("deprecations")
    assert any("demographic_swap_divergence" in msg for msg in body["deprecations"])


def test_contrast_alias_round_trip(assert_no_live_llm_calls):
    cfg = humanitarian_contrast_config()
    raw = {
        "provider": cfg.provider,
        "model": cfg.model,
        "evaluators": ["counterfactual_fairness_contrast"],
        "counterfactual": {
            "template": cfg.counterfactual.template,
            "dimensions": cfg.counterfactual.dimensions,
            "defaults": getattr(cfg.counterfactual, "defaults", {}) or {},
            "name_pools": cfg.counterfactual.name_pools,
            "control_dimension": cfg.counterfactual.control_dimension,
        },
        "cache_dir": cfg.cache_dir,
        "params": cfg.params,
    }
    with pytest.warns(FutureWarning, match="demographic_swap_contrast"):
        loaded = load_llm_eval_config(obj=raw)
    assert loaded.evaluators == [DEMOGRAPHIC_SWAP_CONTRAST]
    result = run_llm_eval(loaded, with_ci=False)
    assert DEMOGRAPHIC_SWAP_CONTRAST in result.metrics
    assert "counterfactual_fairness_contrast" not in result.metrics


def test_unknown_evaluator_still_rejected():
    with pytest.raises(ConfigValidationError, match="Unknown evaluator"):
        load_llm_eval_config(
            obj={
                "provider": "local",
                "model": "demo",
                "evaluators": ["not_a_real_metric"],
            }
        )
