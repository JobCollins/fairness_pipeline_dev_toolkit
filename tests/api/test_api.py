"""
Tests for the fairpipe REST API.

Requires: pip install fairpipe[api]
Skipped automatically when fastapi or httpx is not installed.
"""

from __future__ import annotations

import io
import textwrap
import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="requires fairpipe[api]")
pytest.importorskip("httpx", reason="requires fairpipe[api]")

from fastapi.testclient import TestClient  # noqa: E402

from fairness_pipeline_dev_toolkit import __version__ as _pkg_version  # noqa: E402
from fairness_pipeline_dev_toolkit.api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == _pkg_version
    assert "timestamp" in body


# ---------------------------------------------------------------------------
# POST /validate — happy path
# ---------------------------------------------------------------------------


def test_validate_basic(client):
    payload = {
        "y_pred": [1, 0, 1, 0, 1, 0],
        "sensitive": ["A", "A", "A", "B", "B", "B"],
        "min_group_size": 1,
    }
    r = client.post("/validate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    assert "metrics" in body
    assert "passed" in body
    assert "demographic_parity_difference" in body["metrics"]
    dp = body["metrics"]["demographic_parity_difference"]
    # Same MetricResult.caveat field used by LLM eval provenance; classifier runs are unlabeled.
    assert "caveat" in dp
    assert dp["caveat"] is None


# ---------------------------------------------------------------------------
# POST /validate — mismatched array lengths → 422
# ---------------------------------------------------------------------------


def test_validate_mismatched_lengths(client):
    payload = {
        "y_pred": [1, 0, 1],
        "sensitive": ["A", "B"],  # length mismatch
    }
    r = client.post("/validate", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /validate then GET /results/{run_id}
# ---------------------------------------------------------------------------


def test_validate_stores_result(client):
    payload = {
        "y_pred": [1, 0, 1, 0],
        "sensitive": ["A", "A", "B", "B"],
        "min_group_size": 1,
    }
    r = client.post("/validate", json=payload)
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    r2 = client.get(f"/results/{run_id}")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["run_id"] == run_id


# ---------------------------------------------------------------------------
# GET /results/{run_id} — not found → 404
# ---------------------------------------------------------------------------


def test_results_not_found(client):
    r = client.get("/results/nonexistent-run-id-that-does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# passed=false must return 200, not 500
# ---------------------------------------------------------------------------


def test_validate_passed_false_is_not_500(client):
    # group A always predicted 1, group B always predicted 0 → DPD = 1.0
    payload = {
        "y_pred": [1] * 50 + [0] * 50,
        "sensitive": ["A"] * 50 + ["B"] * 50,
        "threshold": 0.01,  # tight threshold → passed=False
        "min_group_size": 10,
    }
    r = client.post("/validate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False


# ---------------------------------------------------------------------------
# GET /docs → 200 (Swagger UI accessible)
# ---------------------------------------------------------------------------


def test_docs_accessible(client):
    r = client.get("/docs")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Concurrent store — thread safety
# ---------------------------------------------------------------------------


def test_concurrent_store(client):
    results: dict[str, str] = {}
    errors: list[Exception] = []

    def post(key: str) -> None:
        try:
            payload = {
                "y_pred": [1, 0, 1, 0],
                "sensitive": ["A", "A", "B", "B"],
                "min_group_size": 1,
            }
            r = client.post("/validate", json=payload)
            assert r.status_code == 200
            results[key] = r.json()["run_id"]
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=post, args=("t1",))
    t2 = threading.Thread(target=post, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Thread errors: {errors}"
    assert "t1" in results and "t2" in results
    assert results["t1"] != results["t2"]

    r1 = client.get(f"/results/{results['t1']}")
    r2 = client.get(f"/results/{results['t2']}")
    assert r1.status_code == 200
    assert r2.status_code == 200


# ---------------------------------------------------------------------------
# POST /pipeline — multipart CSV + YAML config
# ---------------------------------------------------------------------------

_PIPELINE_CONFIG = textwrap.dedent(
    """
    sensitive: ["sensitive"]
    alpha: 0.05
    pipeline:
      - name: reweigh
        transformer: "InstanceReweighting"
        params: {}
    """
)


def _make_pipeline_csv() -> bytes:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "sensitive": rng.choice(["A", "B"], size=n),
            "score": rng.uniform(0, 1, size=n),
            "y_true": rng.integers(0, 2, size=n),
        }
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_pipeline_basic(client):
    r = client.post(
        "/pipeline",
        files={"file": ("data.csv", _make_pipeline_csv(), "text/csv")},
        data={"config": _PIPELINE_CONFIG},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "run_id" in body
    assert "detector_report" in body
    assert body["transformed_rows"] == 200
    assert len(body["transformers_applied"]) > 0


def test_pipeline_stores_result(client):
    r = client.post(
        "/pipeline",
        files={"file": ("data.csv", _make_pipeline_csv(), "text/csv")},
        data={"config": _PIPELINE_CONFIG},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    r2 = client.get(f"/results/{run_id}")
    assert r2.status_code == 200
    assert r2.json()["run_id"] == run_id


def test_pipeline_invalid_config(client):
    """Config with no pipeline steps → 422."""
    config_no_steps = textwrap.dedent(
        """
        sensitive: ["sensitive"]
        alpha: 0.05
        """
    )
    r = client.post(
        "/pipeline",
        files={"file": ("data.csv", _make_pipeline_csv(), "text/csv")},
        data={"config": config_no_steps},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /workflow — multipart CSV + YAML config
# ---------------------------------------------------------------------------

_WORKFLOW_CONFIG = textwrap.dedent(
    """
    sensitive: ["sensitive"]
    alpha: 0.05
    pipeline:
      - name: reweigh
        transformer: "InstanceReweighting"
        params: {}
    training:
      method: "reductions"
      target_column: "y"
      params:
        constraint: "demographic_parity"
        eps: 0.05
        T: 5
    fairness_metric: "demographic_parity_difference"
    validation_threshold: 0.20
    """
)


def _make_workflow_csv() -> bytes:
    rng = np.random.default_rng(42)
    n = 300
    sensitive = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
    f0 = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    bias = (sensitive == "B").astype(float) * 0.4
    y = ((f0 + f1 + bias + rng.standard_normal(n) * 0.1) > 0).astype(int)
    df = pd.DataFrame({"sensitive": sensitive, "f0": f0, "f1": f1, "y": y})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_workflow_basic(client):
    pytest.importorskip("sklearn", reason="scikit-learn required for workflow")
    r = client.post(
        "/workflow",
        files={"file": ("data.csv", _make_workflow_csv(), "text/csv")},
        data={"config": _WORKFLOW_CONFIG, "min_group_size": "10", "train_size": "0.8"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "run_id" in body
    assert "validation" in body
    assert "passed" in body["validation"]
    assert "baseline_metrics" in body
    assert "final_metrics" in body


def test_workflow_stores_result(client):
    pytest.importorskip("sklearn", reason="scikit-learn required for workflow")
    r = client.post(
        "/workflow",
        files={"file": ("data.csv", _make_workflow_csv(), "text/csv")},
        data={"config": _WORKFLOW_CONFIG, "min_group_size": "10", "train_size": "0.8"},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    r2 = client.get(f"/results/{run_id}")
    assert r2.status_code == 200
    assert r2.json()["run_id"] == run_id
