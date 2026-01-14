"""
Pytest convenience plugin for fairness gating.

Use cases:
- Gate deployments by asserting fairness thresholds in CI pipelines.
- Provide readable failure reports when fairness checks fail.
"""

from __future__ import annotations

import math
from typing import Optional


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
        comparator: Comparison operator as a string. One of "<=", "<", ">=", ">".
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

    ok = (value <= threshold) if comparator == "<=" else (value < threshold)
    if not ok:
        suffix = f" | {context}" if context else ""
        raise AssertionError(
            f"Fairness threshold exceeded: value={value:.6f} comparator={comparator} threshold={threshold:.6f}{suffix}"
        )
