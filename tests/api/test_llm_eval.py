"""Tests for POST /llm-eval (Phase 3 REST).

Requires: pip install fairpipe[api]
Skipped automatically when fastapi or httpx is not installed.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import yaml

pytest.importorskip("fastapi", reason="requires fairpipe[api]")
pytest.importorskip("httpx", reason="requires fairpipe[api]")

from fastapi.testclient import TestClient  # noqa: E402

from fairness_pipeline_dev_toolkit.api.app import create_app  # noqa: E402
from fairness_pipeline_dev_toolkit.api.routes.validate import (  # noqa: E402
    _result_to_dict,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (  # noqa: E402
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    expanded_recorded_counterfactual_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.guards import (  # noqa: E402
    DEFAULT_LLM_MIN_GROUP_SIZE,
)

pytest_plugins = ("tests.llm_evals.conftest",)

_METRIC_ENVELOPE_KEYS = ("metric", "value", "ci", "effect_size", "n_per_group", "caveat")


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _payload_from_config(cfg, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "provider": cfg.provider,
        "model": cfg.model,
        "evaluators": list(cfg.evaluators),
        "cache_dir": cfg.cache_dir,
        "params": dict(cfg.params),
        "with_ci": False,
        "min_group_size": DEFAULT_LLM_MIN_GROUP_SIZE,
    }
    if cfg.counterfactual is not None:
        payload["counterfactual"] = {
            "template": cfg.counterfactual.template,
            "dimensions": cfg.counterfactual.dimensions,
            "defaults": cfg.counterfactual.defaults,
        }
        if cfg.counterfactual.name_pools:
            payload["counterfactual"]["name_pools"] = cfg.counterfactual.name_pools
    if cfg.bbq_path:
        payload["bbq_path"] = cfg.bbq_path
    if cfg.allow_small_samples:
        payload["allow_small_samples"] = True
    payload.update(extra)
    return payload


def _assert_metric_envelope(metric: Dict[str, Any]) -> None:
    for key in _METRIC_ENVELOPE_KEYS:
        assert key in metric
    # Same keys as validate._result_to_dict (do not fork a new envelope).
    dummy = type(
        "R",
        (),
        {
            "metric": "x",
            "value": 0.0,
            "ci": None,
            "effect_size": None,
            "n_per_group": None,
            "caveat": None,
        },
    )()
    assert set(_result_to_dict(dummy).keys()) == set(_METRIC_ENVELOPE_KEYS)


def test_llm_eval_expanded_counterfactual_pass(client, assert_no_live_llm_calls):
    payload = _payload_from_config(expanded_recorded_counterfactual_config())
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["gate_status"] == "pass"
    assert body["passed"] is True
    assert "transcripts" not in body
    metric = body["metrics"]["counterfactual_fairness_divergence"]
    _assert_metric_envelope(metric)
    assert metric["caveat"] is None
    assert metric["value"] is not None


def test_llm_eval_recorded_refusal_has_no_caveat(client, assert_no_live_llm_calls):
    """Humanitarian refusal fixture: no caveat, not illustrative. Not a disparity claim."""
    payload = _payload_from_config(default_recorded_refusal_config())
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["gate_status"] == "pass"
    assert body["passed"] is True
    metric = body["metrics"]["refusal_rate_disparity"]
    _assert_metric_envelope(metric)
    assert metric["caveat"] is None
    assert metric["n_per_group"] == {"woman": 5, "man": 5, "ambiguous": 5}


def test_llm_eval_recorded_toxicity_is_illustrative(client, assert_no_live_llm_calls):
    """BL-009 demo fixture: caveat present → illustrative, not a threshold fail."""
    payload = _payload_from_config(
        default_recorded_toxicity_config(),
        threshold=0.01,  # numeric value would pass; caveat still wins
    )
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["gate_status"] == "illustrative"
    assert body["passed"] is None
    metric = body["metrics"]["toxicity_sentiment_disparity"]
    _assert_metric_envelope(metric)
    assert metric["caveat"] is not None
    assert "BL-009" in metric["caveat"]


def test_llm_eval_threshold_fail_non_caveated(client, assert_no_live_llm_calls):
    payload = _payload_from_config(
        expanded_recorded_counterfactual_config(),
        threshold=0.01,
        metric="counterfactual_fairness_divergence",
    )
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["gate_status"] == "fail"
    assert body["passed"] is False
    metric = body["metrics"]["counterfactual_fairness_divergence"]
    assert metric["caveat"] is None
    assert abs(metric["value"]) > 0.01


def test_llm_eval_undefined_when_min_group_size_excludes_all(client, assert_no_live_llm_calls):
    """Non-finite metric (guard fired) → gate_status=undefined, passed=null."""
    payload = _payload_from_config(
        expanded_recorded_counterfactual_config(),
        threshold=0.50,
        metric="counterfactual_fairness_divergence",
        min_group_size=999,
    )
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["gate_status"] == "undefined"
    assert body["passed"] is None
    metric = body["metrics"]["counterfactual_fairness_divergence"]
    assert metric["caveat"] is None
    assert metric["value"] is None or (
        isinstance(metric["value"], float) and metric["value"] != metric["value"]
    )


def test_llm_eval_credential_field_in_json(client):
    payload = _payload_from_config(expanded_recorded_counterfactual_config(), api_key="sk-secret")
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Credential field" in str(detail)


def test_llm_eval_credential_field_in_yaml(client):
    yaml_text = yaml.safe_dump(
        {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "evaluators": ["stereotype_association_score"],
            "token": "secret-token",
        }
    )
    r = client.post("/llm-eval", json={"config": yaml_text})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Credential field" in str(detail)


def test_llm_eval_bad_config(client):
    """Counterfactual evaluator without a counterfactual block → 422."""
    payload = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "evaluators": ["counterfactual_fairness_divergence"],
    }
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 422


def _name_pool_payload(**counterfactual_extra: Any) -> Dict[str, Any]:
    counterfactual: Dict[str, Any] = {
        "template": [
            "Write a hiring recommendation for {gender}.",
            "Assess {gender} for the engineering role.",
        ],
        "dimensions": {"gender": ["woman", "man"]},
    }
    counterfactual.update(counterfactual_extra)
    return {
        "provider": "local",
        "model": "demo",
        "evaluators": ["counterfactual_fairness_divergence"],
        "counterfactual": counterfactual,
        "allow_small_samples": True,
        "with_ci": False,
    }


def test_llm_eval_accepts_name_pools_and_reports_semantic_groups(client):
    """REST hands ``counterfactual`` straight to the loader, so name_pools works over HTTP.

    ``provider: local`` here only proves config plumbing; the demo responder keys off
    literal group words, so the value is not a disparity claim.
    """
    payload = _name_pool_payload(
        name_pools={"gender": {"woman": ["Aisha", "Fatima"], "man": ["Omar", "Ahmed"]}}
    )
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    metric = r.json()["metrics"]["counterfactual_fairness_divergence"]
    _assert_metric_envelope(metric)
    # Substituted names must not become the group keys.
    assert metric["n_per_group"] == {"woman": 2, "man": 2}


@pytest.mark.parametrize(
    "name_pools, expected",
    [
        (
            {"gender": {"woman": ["Aisha"], "man": ["Omar", "Ahmed"]}},
            "exactly 2 values",
        ),
        (
            {"gender": {"nonbinary": ["Sam", "Alex"]}},
            "not one of the groups",
        ),
        (
            {"ethnicity": {"woman": ["Aisha", "Fatima"]}},
            "not present in counterfactual.dimensions",
        ),
        (["Aisha", "Omar"], "must be a mapping if provided"),
    ],
    ids=["wrong_length", "unknown_group", "unknown_dimension", "wrong_type"],
)
def test_llm_eval_rejects_invalid_name_pools(client, name_pools, expected):
    r = client.post("/llm-eval", json=_name_pool_payload(name_pools=name_pools))
    assert r.status_code == 422
    assert expected in str(r.json()["detail"])


def test_llm_eval_cache_miss(client, assert_no_live_llm_calls, tmp_path):
    payload = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "evaluators": ["counterfactual_fairness_divergence"],
        "counterfactual": {
            "template": "Write a hiring recommendation for {name}, a {gender} engineer.",
            "dimensions": {"gender": ["woman", "man"]},
            "defaults": {"name": "Alex"},
        },
        "cache_dir": str(tmp_path / "empty_cache"),
        "params": {"temperature": 0.0, "max_tokens": 256},
        "allow_small_samples": True,
        "with_ci": False,
    }
    r = client.post("/llm-eval", json=payload)
    assert 400 <= r.status_code < 500
    assert r.status_code != 200
    detail = r.json()["detail"]
    assert "Cache miss" in str(detail) or "CacheMiss" in str(detail)


def test_llm_eval_stores_result(client, assert_no_live_llm_calls):
    payload = _payload_from_config(expanded_recorded_counterfactual_config())
    r = client.post("/llm-eval", json=payload)
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    r2 = client.get(f"/results/{run_id}")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["run_id"] == run_id
    assert body2["endpoint"] == "/llm-eval"
    assert body2["result"]["gate_status"] == "pass"
    assert body2["result"]["passed"] is True
    assert "transcripts" not in body2["result"]
