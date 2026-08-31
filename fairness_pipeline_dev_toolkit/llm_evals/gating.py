"""Three-state LLM-eval gate: pass / fail / illustrative.

``gate_status`` is canonical. ``passed`` is aligned 1:1 so a bool-only client
does not treat an illustrative (demo-fixture) result as a threshold fail:

    pass         → passed=True
    fail         → passed=False   (threshold miss on a non-caveated metric)
    illustrative  → passed=None    (gated metric has a non-null caveat)

CLI / Action harness (not HTTP) maps: pass=0, fail=1, usage=2, illustrative=3.
REST returns HTTP 200 for all three gate outcomes, matching ``/validate``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

GateStatus = Literal["pass", "fail", "illustrative"]

GATE_PASS: GateStatus = "pass"
GATE_FAIL: GateStatus = "fail"
GATE_ILLUSTRATIVE: GateStatus = "illustrative"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_ILLUSTRATIVE = 3

GATE_STATUS_TO_EXIT: Dict[GateStatus, int] = {
    GATE_PASS: EXIT_PASS,
    GATE_FAIL: EXIT_FAIL,
    GATE_ILLUSTRATIVE: EXIT_ILLUSTRATIVE,
}

PASSED_BY_GATE: Dict[GateStatus, Optional[bool]] = {
    GATE_PASS: True,
    GATE_FAIL: False,
    GATE_ILLUSTRATIVE: None,
}


def _metric_caveat(item: Any) -> Optional[str]:
    if isinstance(item, Mapping):
        caveat = item.get("caveat")
    else:
        caveat = getattr(item, "caveat", None)
    if caveat is None:
        return None
    text = str(caveat).strip()
    return text or None


def _metric_value(item: Any) -> Optional[float]:
    if isinstance(item, Mapping):
        raw = item.get("value")
    else:
        raw = getattr(item, "value", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def evaluate_llm_eval_gate(
    metrics: Mapping[str, Any],
    *,
    threshold: Optional[float] = None,
    metric: Optional[str] = None,
) -> Tuple[GateStatus, Optional[bool]]:
    """Return ``(gate_status, passed)`` for an LLM eval metrics mapping.

    If ``metric`` is set, only that result is inspected. Otherwise every
    returned metric is inspected. A non-null caveat on the gated metric(s)
    is always ``illustrative``, even when the number would pass ``threshold``.
    """
    if metric is not None:
        if metric not in metrics:
            raise KeyError(metric)
        selected_items = [metrics[metric]]
    else:
        selected_items = list(metrics.values())

    if any(_metric_caveat(item) for item in selected_items):
        return GATE_ILLUSTRATIVE, PASSED_BY_GATE[GATE_ILLUSTRATIVE]

    if threshold is not None:
        for item in selected_items:
            value = _metric_value(item)
            if value is not None and abs(value) > threshold:
                return GATE_FAIL, PASSED_BY_GATE[GATE_FAIL]

    return GATE_PASS, PASSED_BY_GATE[GATE_PASS]
