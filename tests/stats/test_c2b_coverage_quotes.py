"""Pin published C2b coverage quotes to ``tmin_c2b_summary.json``.

Regenerate the quoted strings with::

    .venv/bin/python investigations/wave3a/coverage_quotes.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUOTES_PY = ROOT / "investigations" / "wave3a" / "coverage_quotes.py"
SUMMARY = ROOT / "investigations" / "wave3a" / "results" / "tmin_c2b_summary.json"


def _load_quotes_mod():
    spec = importlib.util.spec_from_file_location("wave3a_coverage_quotes", QUOTES_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def quotes():
    if not SUMMARY.is_file():
        pytest.skip("tmin_c2b_summary.json not present")
    return _load_quotes_mod()


def test_c2b_display_matches_user_table(quotes):
    """Half-up 3 d.p. display must match the committed summary."""
    q = quotes.format_c2b_by_T()
    assert q["3"] == {"mean": "0.790", "worst": "0.736"}
    assert q["4"] == {"mean": "0.875", "worst": "0.845"}
    assert q["5"] == {"mean": "0.929", "worst": "0.920"}
    assert q["7"] == {"mean": "0.944", "worst": "0.934"}
    assert q["10"] == {"mean": "0.945", "worst": "0.939"}


def test_docs_and_module_quote_summary_json(quotes):
    """CHANGELOG / api.md / template_intervals Notes must not drift from JSON."""
    q = quotes.format_c2b_by_T()
    rst = quotes.c2b_rst_table()
    md = quotes.c2b_markdown_table()

    template_mod = (
        ROOT / "fairness_pipeline_dev_toolkit" / "stats" / "template_intervals.py"
    ).read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    assert rst in template_mod
    assert f"T=5 / 7 / 10 was {q['5']['mean']} / {q['7']['mean']} / {q['10']['mean']}" in (
        template_mod
    )
    assert f"(worst {q['5']['worst']} / {q['7']['worst']} / {q['10']['worst']})" in template_mod
    assert md in api
    # CHANGELOG may wrap lines; require every T's mean/worst pair.
    for T in ("3", "4", "5", "7", "10"):
        assert q[T]["mean"] in changelog and q[T]["worst"] in changelog
        assert q[T]["mean"] in api and q[T]["worst"] in api
        assert q[T]["mean"] in template_mod and q[T]["worst"] in template_mod
