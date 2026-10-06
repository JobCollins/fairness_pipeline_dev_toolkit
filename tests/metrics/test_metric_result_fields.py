"""Public API field contract: Result ≡ MetricResult."""

from __future__ import annotations

from dataclasses import fields

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult
from fairness_pipeline_dev_toolkit.metrics.core import Result


def test_result_fields_match_metric_result() -> None:
    mr = [f.name for f in fields(MetricResult)]
    rr = [f.name for f in fields(Result)]
    assert rr == mr
    assert "caveat" in rr
    assert issubclass(Result, MetricResult)
