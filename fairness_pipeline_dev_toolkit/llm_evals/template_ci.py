"""Build template-level arm values for Bonferroni-t LLM intervals (BL-016)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .probes.counterfactual import (
    CounterfactualPrompt,
    extract_response_features,
    pairwise_divergence,
    response_key,
)


def _pair_divergence(
    left: CounterfactualPrompt,
    right: CounterfactualPrompt,
    responses: Dict[Tuple[str, str, int], str],
) -> float:
    left_text = responses[response_key(left)]
    right_text = responses[response_key(right)]
    feat_left = extract_response_features(left_text, reference=right_text)
    feat_right = extract_response_features(right_text, reference=left_text)
    return pairwise_divergence(feat_left, feat_right)


def _template_mean_for_dimension(
    items: List[CounterfactualPrompt],
    responses: Dict[Tuple[str, str, int], str],
    required_groups: Set[str],
) -> Optional[float]:
    """Mean pairwise divergence within one (dimension, replicate) bucket, or None if incomplete."""
    by_group = {p.group: p for p in items}
    if not required_groups.issubset(by_group):
        return None
    groups = sorted(required_groups)
    vals: List[float] = []
    for i, g in enumerate(groups):
        for h in groups[i + 1 :]:
            vals.append(_pair_divergence(by_group[g], by_group[h], responses))
    return float(sum(vals) / len(vals)) if vals else None


def divergence_template_arms(
    prompts: Sequence[CounterfactualPrompt],
    responses: Dict[Tuple[str, str, int], str],
    *,
    dimensions: Sequence[str],
) -> Tuple[Dict[str, List[float]], int]:
    """Per-dimension template means for divergence Bonferroni-t.

    Only templates complete for *every* listed dimension (all groups present in
    each) are kept; every arm then has the same length ``T``.

    Returns
    -------
    arms, T
        ``arms[dim]`` is a length-T list of per-template means; ``T`` is that length.
    """
    dims = list(dimensions)
    if not dims:
        return {}, 0

    # groups that appear in each dimension (union of observed groups)
    groups_by_dim: Dict[str, Set[str]] = {d: set() for d in dims}
    buckets: Dict[Tuple[str, int], List[CounterfactualPrompt]] = {}
    for p in prompts:
        if p.dimension not in groups_by_dim:
            continue
        groups_by_dim[p.dimension].add(p.group)
        buckets.setdefault((p.dimension, p.replicate_id), []).append(p)

    # Candidate replicate_ids that appear in every dimension.
    reps_by_dim = {d: {rid for (dim, rid) in buckets if dim == d} for d in dims}
    complete_reps = sorted(set.intersection(*(reps_by_dim[d] for d in dims))) if dims else []

    arms: Dict[str, List[float]] = {d: [] for d in dims}
    kept: List[int] = []
    for rid in complete_reps:
        means: Dict[str, float] = {}
        ok = True
        for d in dims:
            m = _template_mean_for_dimension(buckets[(d, rid)], responses, groups_by_dim[d])
            if m is None:
                ok = False
                break
            means[d] = m
        if not ok:
            continue
        kept.append(rid)
        for d in dims:
            arms[d].append(means[d])
    return arms, len(kept)


def contrast_template_arms(
    prompts: Sequence[CounterfactualPrompt],
    responses: Dict[Tuple[str, str, int], str],
    *,
    gated_dimensions: Sequence[str],
    control_dimension: str,
) -> Tuple[Dict[str, List[float]], int]:
    """Per-gated-dimension template contrasts (gated − control) for contrast Bonferroni-t.

    A template is kept only when it is complete for the control dimension and
    every gated dimension. Each arm value is
    ``mean_pairs(gated, r) − mean_pairs(control, r)``.
    """
    dims = list(gated_dimensions)
    all_dims = [*dims, control_dimension]
    groups_by_dim: Dict[str, Set[str]] = {d: set() for d in all_dims}
    buckets: Dict[Tuple[str, int], List[CounterfactualPrompt]] = {}
    for p in prompts:
        if p.dimension not in groups_by_dim:
            continue
        groups_by_dim[p.dimension].add(p.group)
        buckets.setdefault((p.dimension, p.replicate_id), []).append(p)

    reps_by_dim = {d: {rid for (dim, rid) in buckets if dim == d} for d in all_dims}
    complete_reps = sorted(set.intersection(*(reps_by_dim[d] for d in all_dims)))

    arms: Dict[str, List[float]] = {d: [] for d in dims}
    for rid in complete_reps:
        control_mean = _template_mean_for_dimension(
            buckets[(control_dimension, rid)], responses, groups_by_dim[control_dimension]
        )
        if control_mean is None:
            continue
        means: Dict[str, float] = {}
        ok = True
        for d in dims:
            m = _template_mean_for_dimension(buckets[(d, rid)], responses, groups_by_dim[d])
            if m is None:
                ok = False
                break
            means[d] = m - control_mean
        if not ok:
            continue
        for d in dims:
            arms[d].append(means[d])
    T = len(next(iter(arms.values()))) if arms and dims else 0
    # Drop empty arms (no gated pairs) — should not happen if dims non-empty.
    arms = {d: v for d, v in arms.items() if v}
    return arms, T
