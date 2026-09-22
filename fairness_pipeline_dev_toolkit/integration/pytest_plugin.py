"""
Pytest convenience plugin for fairness gating.

Use cases:
- Gate deployments by asserting fairness thresholds in CI pipelines.
- Provide readable failure reports when fairness checks fail.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    GATE_FAIL,
    GATE_ILLUSTRATIVE,
    GATE_PASS,
    GATE_UNDEFINED,
    evaluate_llm_eval_gate,
)
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


def assert_fairness(
    value: float,
    threshold: float,
    *,
    comparator: str = "<=",
    allow_nan: bool = False,
    context: Optional[str] = None,
) -> None:
    """
    Assert that a fairness metric meets the specified threshold.

    Args:
        value: The computed fairness metric value e.g demographic parity difference.
        threshold: maximum allowed disparity e.g. 0.10 for 10%.
        comparator: Comparison operator as a string: "<=", "<", ">=", or ">".
        allow_nan: If True and value is Nan, the check passes silently. (useful when groups are too small)
        context: Optional context string to include in failure messages.

    Raises:
        AssertionError: If the fairness check fails.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if allow_nan:
            return
        suffix = f" | {context}" if context else ""
        raise AssertionError(f"Fairness metric is NaN (insufficient data?){suffix}")

    if comparator == "<=":
        ok = value <= threshold
    elif comparator == "<":
        ok = value < threshold
    elif comparator == ">=":
        ok = value >= threshold
    elif comparator == ">":
        ok = value > threshold
    else:
        raise ValueError(f"comparator must be one of '<=', '<', '>=', '>'; got {comparator!r}")

    if not ok:
        suffix = f" | {context}" if context else ""
        raise AssertionError(
            f"Fairness threshold exceeded: value={value:.6f} comparator={comparator} threshold={threshold:.6f}{suffix}"
        )


def _as_metric_result(value: Any) -> MetricResult:
    """Normalize a scalar or ``MetricResult`` for ``evaluate_llm_eval_gate``."""
    if isinstance(value, MetricResult):
        return value
    if isinstance(value, dict) and "value" in value:
        return MetricResult(
            metric=str(value.get("metric") or "llm_eval"),
            value=float(value["value"]) if value["value"] is not None else float("nan"),
            caveat=value.get("caveat"),
        )
    # Bare float / None — no caveat (same as pre-fix scalar path, minus comparator).
    raw = value
    if raw is None:
        raw = float("nan")
    return MetricResult(metric="llm_eval", value=float(raw), caveat=None)


def assert_llm_fairness(
    value: Any,
    threshold: float,
    *,
    comparator: str = "<=",
    allow_nan: bool = False,
    context: Optional[str] = None,
) -> None:
    """
    Assert that an LLM fairness metric meets the CLI / REST gate policy.

    Routes through :func:`~fairness_pipeline_dev_toolkit.llm_evals.gating.evaluate_llm_eval_gate`
    so pytest, CLI, and Action share one rule:

    - Non-null ``MetricResult.caveat`` → fail as **illustrative** (CLI exit 3), even when
      the number would pass ``threshold``. Precedence: illustrative wins over undefined.
    - Non-finite ``value`` (NaN / inf) → fail as **undefined** (CLI exit 4). This usually
      means ``min_group_size`` excluded every eligible group. With ``allow_nan=True``,
      undefined is tolerated (plugin-only opt-in; CLI / REST still report undefined).
    - Otherwise fail when ``abs(value) > threshold`` (magnitude-based; signed metrics such
      as ``demographic_swap_contrast`` are gated on absolute size).

    ``comparator`` is retained for call-site compatibility with :func:`assert_fairness`
    but is **not** applied. Classifier checks still go through :func:`assert_fairness`.

    Args:
        value: A ``MetricResult`` (preferred) or a scalar float.
        threshold: Maximum allowed ``abs(value)``.
        comparator: Ignored (deprecated for LLM gates).
        allow_nan: If True, skip the raise when the gate status is ``undefined``
            (insufficient evidence / ``min_group_size``). Default False restores the
            pre-Wave-1c raise-on-NaN behaviour.
        context: Optional suffix for failure messages.

    Raises:
        AssertionError: On illustrative caveat, undefined metric, or magnitude miss.
    """
    del comparator  # not part of the shared LLM gate contract
    result = _as_metric_result(value)
    name = result.metric or "llm_eval"
    gate_status, _passed = evaluate_llm_eval_gate(
        {name: result},
        threshold=threshold,
        metric=name,
    )
    suffix = f" | {context}" if context else ""

    if gate_status == GATE_ILLUSTRATIVE:
        caveat = (result.caveat or "").strip() or "(no caveat text)"
        raise AssertionError(
            "LLM fairness result is illustrative (CLI exit 3): gated metric has a "
            f"non-null caveat — even if the number would pass the threshold. "
            f"caveat={caveat!r} value={result.value!r} threshold={threshold:.6f}{suffix}"
        )
    if gate_status == GATE_UNDEFINED:
        if allow_nan:
            return
        raise AssertionError(
            "LLM fairness metric is undefined (CLI exit 4): insufficient evidence — "
            "a group likely fell below min_group_size, so no finite disparity number "
            f"was produced. value={result.value!r} threshold={threshold:.6f}{suffix}"
        )
    if gate_status == GATE_FAIL:
        raise AssertionError(
            f"LLM fairness threshold exceeded (CLI exit 1): "
            f"abs(value)={abs(float(result.value)):.6f} > threshold={threshold:.6f} "
            f"(value={result.value!r}; gating is magnitude-based){suffix}"
        )
    if gate_status != GATE_PASS:
        raise AssertionError(f"Unexpected LLM gate status {gate_status!r}{suffix}")
