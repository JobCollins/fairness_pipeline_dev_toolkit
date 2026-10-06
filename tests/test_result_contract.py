"""Wave 3a result contract: ``p_value`` / ``ci_kind`` / ``ci_note`` through every serializer."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, fields

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.api.routes.validate import _result_to_dict
from fairness_pipeline_dev_toolkit.api.routes.workflow import _serialize_metrics
from fairness_pipeline_dev_toolkit.integration.pytest_plugin import _as_metric_result
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer, MetricResult
from fairness_pipeline_dev_toolkit.metrics.core import Result

NEW_FIELDS = ("p_value", "ci_kind", "ci_note")


def _from_dict(cls, d):
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in names}
    if kwargs.get("ci") is not None:
        kwargs["ci"] = tuple(kwargs["ci"])
    return cls(**kwargs)


@pytest.fixture
def analyzer_result() -> Result:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.4, 120)
    s = np.repeat(["a", "b", "c"], 40)
    return FairnessAnalyzer(min_group_size=30, backend="native").demographic_parity_difference(y, s)


@pytest.mark.parametrize("cls", [Result, MetricResult])
def test_new_fields_default_none_and_old_payloads_load(cls):
    assert [f.name for f in fields(cls)][-3:] == list(NEW_FIELDS)
    old = {"metric": "m", "value": 0.1, "ci": [0.0, 0.2], "effect_size": None}
    r = _from_dict(cls, old)
    assert all(getattr(r, k) is None for k in NEW_FIELDS)
    positional = cls("m", 0.1, (0.0, 0.2), None, {"a": 1}, None, 0)
    assert positional.p_value is None


def test_analyzer_result_populates_contract(analyzer_result):
    r = analyzer_result
    assert r.ci_kind == "simultaneous_pairwise"
    assert r.ci_note is None
    assert 0 < r.p_value <= 1


def test_rest_round_trip(analyzer_result):
    payload = json.loads(json.dumps(_result_to_dict(analyzer_result)))
    for k in NEW_FIELDS:
        assert k in payload
    back = _from_dict(Result, payload)
    assert back.ci == pytest.approx(analyzer_result.ci)
    assert back.p_value == pytest.approx(analyzer_result.p_value)
    assert back.ci_kind == analyzer_result.ci_kind
    assert back.ci_note == analyzer_result.ci_note


def test_rest_undefined_ci_serializes_as_null():
    r = Result("m", 0.3, ci=(float("nan"), float("nan")), ci_note="undefined:x")
    d = _result_to_dict(r)
    assert d["ci"] is None and d["ci_note"] == "undefined:x" and d["p_value"] is None


def test_rest_metric_result_round_trip():
    m = MetricResult(
        "demographic_swap_divergence",
        0.2,
        ci=(0.1, 0.3),
        ci_kind="template_bonferroni_t",
        ci_note="small_T: T=6 templates",
    )
    back = _from_dict(MetricResult, json.loads(json.dumps(_result_to_dict(m))))
    assert back.ci == (0.1, 0.3)
    assert back.ci_kind == m.ci_kind and back.ci_note == m.ci_note and back.p_value is None


def test_workflow_serializer_uses_same_envelope(analyzer_result):
    out = _serialize_metrics({"dpd": analyzer_result, "acc": 0.9})
    assert out["dpd"] == _result_to_dict(analyzer_result)
    assert out["acc"] == 0.9


def test_asdict_json_round_trip(analyzer_result):
    """Reporting JSON and the MLflow artifact both serialize via dataclasses.asdict."""
    back = _from_dict(Result, json.loads(json.dumps(asdict(analyzer_result))))
    assert back == analyzer_result


def test_pytest_plugin_dict_round_trip(analyzer_result):
    m = _as_metric_result(_result_to_dict(analyzer_result))
    assert m.ci == pytest.approx(analyzer_result.ci)
    assert (m.p_value, m.ci_kind, m.ci_note) == pytest.approx(
        (analyzer_result.p_value, analyzer_result.ci_kind, analyzer_result.ci_note)
    )


def test_mlflow_logs_new_fields(monkeypatch, analyzer_result):
    calls = {"metric": {}, "param": {}, "dict": {}}
    fake = types.ModuleType("mlflow")
    fake.log_metric = lambda k, v: calls["metric"].__setitem__(k, v)
    fake.log_param = lambda k, v: calls["param"].__setitem__(k, v)
    fake.log_dict = lambda d, name: calls["dict"].__setitem__(name, d)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    from fairness_pipeline_dev_toolkit.integration.mlflow_logger import (
        log_fairness_metrics,
    )

    assert log_fairness_metrics({"dpd": analyzer_result}) is True
    assert calls["metric"]["fairness_.dpd.p_value"] == pytest.approx(analyzer_result.p_value)
    assert json.loads(calls["param"]["fairness_.dpd.ci_kind"]) == "simultaneous_pairwise"
    assert "fairness_.dpd.ci_note" not in calls["param"]
    artifact = json.loads(json.dumps(calls["dict"]["fairness_results.json"]))
    assert _from_dict(Result, artifact["dpd"]) == analyzer_result
