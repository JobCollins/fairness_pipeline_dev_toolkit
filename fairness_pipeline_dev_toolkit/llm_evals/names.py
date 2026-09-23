"""Canonical LLM-eval metric names and deprecated aliases.

All entry points (YAML config, REST, CLI ``--metric``, pytest plugin / gate)
resolve aliases through :func:`canonicalize_evaluator_name` so they cannot
disagree. Output keys (``result.metrics``, reports, REST, MLflow) use only the
canonical names — never both.

Alias notices use :class:`FutureWarning` (visible under default filters; unlike
:class:`DeprecationWarning`, which is often hidden for library code).
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Sequence

# Canonical (output) names
DEMOGRAPHIC_SWAP_DIVERGENCE = "demographic_swap_divergence"
DEMOGRAPHIC_SWAP_CONTRAST = "demographic_swap_contrast"

# Deprecated input aliases → canonical
EVALUATOR_ALIASES: dict[str, str] = {
    "counterfactual_fairness_divergence": DEMOGRAPHIC_SWAP_DIVERGENCE,
    "counterfactual_fairness_contrast": DEMOGRAPHIC_SWAP_CONTRAST,
}

# Primary names listed in docs / VALID_EVALUATORS (aliases are accepted but not listed)
VALID_EVALUATORS = frozenset(
    {
        DEMOGRAPHIC_SWAP_DIVERGENCE,
        DEMOGRAPHIC_SWAP_CONTRAST,
        "refusal_rate_disparity",
        "toxicity_sentiment_disparity",
        "stereotype_association_score",
    }
)

ACCEPTED_EVALUATORS = VALID_EVALUATORS | frozenset(EVALUATOR_ALIASES)

# Evaluators that require a ``counterfactual:`` YAML / REST config block
NEEDS_COUNTERFACTUAL_BLOCK = frozenset(
    {
        DEMOGRAPHIC_SWAP_DIVERGENCE,
        DEMOGRAPHIC_SWAP_CONTRAST,
        "refusal_rate_disparity",
        "toxicity_sentiment_disparity",
    }
)

# Shared prompt set for dry-run costing (counted once even if both listed)
DEMOGRAPHIC_SWAP_PROBE_EVALS = frozenset(
    {
        DEMOGRAPHIC_SWAP_DIVERGENCE,
        DEMOGRAPHIC_SWAP_CONTRAST,
    }
)

_DEPRECATION_TEMPLATE = (
    "{old!r} is deprecated and will be removed after one release; use {new!r} instead."
)


def alias_deprecation_message(name: str) -> Optional[str]:
    """Return the human-readable alias notice for ``name``, or ``None`` if canonical."""
    canonical = EVALUATOR_ALIASES.get(name)
    if canonical is None:
        return None
    return _DEPRECATION_TEMPLATE.format(old=name, new=canonical)


def collect_alias_deprecations(names: Iterable[str]) -> List[str]:
    """Return deprecation messages for any deprecated names in ``names`` (deduped)."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in names:
        text = str(raw)
        if text in seen:
            continue
        seen.add(text)
        msg = alias_deprecation_message(text)
        if msg is not None:
            out.append(msg)
    return out


def canonicalize_evaluator_name(name: str, *, warn: bool = True) -> str:
    """Map a deprecated alias to its canonical name; pass through unknowns/canonicals.

    Emits :class:`FutureWarning` when ``name`` is an alias and ``warn`` is True
    (visible under default warning filters).
    Does not validate membership in :data:`VALID_EVALUATORS` — callers that need
    validation should check :data:`ACCEPTED_EVALUATORS` / :data:`VALID_EVALUATORS`.
    """
    msg = alias_deprecation_message(name)
    if msg is None:
        return name
    if warn:
        warnings.warn(msg, FutureWarning, stacklevel=2)
    return EVALUATOR_ALIASES[name]


def canonicalize_evaluators(
    names: Sequence[str] | Iterable[str],
    *,
    warn: bool = True,
) -> List[str]:
    """Canonicalize a list of evaluator names (preserving order, de-duplicating)."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in names:
        canonical = canonicalize_evaluator_name(str(raw), warn=warn)
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out
