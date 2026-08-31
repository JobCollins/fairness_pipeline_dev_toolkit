"""
Shared min_group_size filtering for LLM fairness evals.

Mirrors ``NativeAdapter._group_mask`` / ``FairnessAnalyzer`` semantics:
groups with fewer than ``min_group_size`` samples are excluded silently;
metrics return ``nan`` when fewer than two eligible groups remain. No warnings
or raises — identical failure mode to classifier fairness metrics.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Deliberate default below classifier ``FairnessAnalyzer``'s 30: each LLM eval
# sample is a paid provider API call. Five prompts per group is the minimum
# pilot size that still supports bootstrap CI in typical probe designs; use
# ``allow_small_samples=True`` only for illustrative / smoke-test runs.
DEFAULT_LLM_MIN_GROUP_SIZE = 5


def apply_min_group_size(
    n_per_group: Dict[str, int],
    min_group_size: int,
    *,
    allow_small_samples: bool = False,
) -> Tuple[Dict[str, int], bool]:
    """
    Filter per-group prompt counts using classifier-style exclusion.

    Returns ``(eligible_n_per_group, can_compute)`` where *can_compute* is
    ``True`` when at least two groups meet the threshold (or when
    *allow_small_samples* bypasses the threshold but two or more groups exist).

    When no group meets the threshold, *eligible_n_per_group* is empty — same
    as ``NativeAdapter`` returning ``n_per_group={}`` when ``valid.sum() == 0``.
    """
    if allow_small_samples:
        return dict(n_per_group), len(n_per_group) >= 2

    eligible = {group: count for group, count in n_per_group.items() if count >= min_group_size}
    return eligible, len(eligible) >= 2


def filter_items_by_eligible_groups(items, *, group_attr: str, eligible_groups: Dict[str, int]):
    """Keep probe rows whose group key is in *eligible_groups*."""
    eligible = set(eligible_groups)
    return [item for item in items if getattr(item, group_attr) in eligible]
