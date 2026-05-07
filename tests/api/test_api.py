"""
Tests for the fairpipe REST API.

Requires: pip install fairpipe[api]
Skipped automatically when fastapi or httpx is not installed.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("fastapi", reason="requires fairpipe[api]")
pytest.importorskip("httpx", reason="requires fairpipe[api]")

from fastapi.testclient import TestClient  # noqa: E402

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
    assert body["version"] == "0.7.0"
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
