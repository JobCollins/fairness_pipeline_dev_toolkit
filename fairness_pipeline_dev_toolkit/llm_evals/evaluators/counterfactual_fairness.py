"""Deprecated module path — use :mod:`demographic_swap`."""

from __future__ import annotations

import warnings

from .demographic_swap import DemographicSwapEvaluator


class CounterfactualFairnessEvaluator(DemographicSwapEvaluator):
    """Deprecated alias for :class:`DemographicSwapEvaluator`."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        warnings.warn(
            "CounterfactualFairnessEvaluator is deprecated; use DemographicSwapEvaluator.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["CounterfactualFairnessEvaluator", "DemographicSwapEvaluator"]
