from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ..utils.validation import coerce_arrays, check_lengths
from .base import MetricResult
from .native_adapter import NativeAdapter
from .fairlearn_adapter import FairlearnAdapter
from .aequitas_adapter import AequitasAdapter


@dataclass
class Result:
    metric: str
    value: float
    ci: Optional[tuple[float, float]] = None
    effect_size: Optional[float] = None
    n_per_group: Optional[Dict[str, int]] = None

class FairnessAnalyzer:
    """
    User-facing orchestrator.
    - Maintains an adapter registry.
    - Auto-selects first available backend unless user specifies 'backend'.
    - Provides a tiny per-call cache for group computations (fast, simple).
    """

    def __init__(self, *, min_group_size: int = 30, nan_policy: str = "exclude", backend: Optional[str] = None):
        self.min_group_size = min_group_size
        self.nan_policy = nan_policy

        # adapter registry (order matters for auto-selection)
        self._adapters: Dict[str, Any] = {
            "fairlearn": FairlearnAdapter(),
            "aequitas": AequitasAdapter(),
            "native": NativeAdapter(),
        }

        if backend is None:
            # choose first available adapter in the registry order
            for name, adapter in self._adapters.items():
                if adapter.available():
                    self._backend = name
                    self._adapter = adapter
                    break
        else:
            if backend not in self._adapters:
                raise ValueError(f"Requested backend '{backend}' is not recognized.")
            adapter = self._adapters[backend]
            if not adapter.available():
                raise RuntimeError(f"Requested backend '{backend}' is not available.")
            self._backend = backend
            self._adapter = adapter

        #simple per-instance cache for call-scoped data
        self._cache: Dict[str, Any] = {}

    def _to_result(self, mr: MetricResult) -> Result:
        # Normalize adapter outputs to our Result dataclass for backward compatibility
        return Result(
            metric=mr.metric,
            value=mr.value,
            ci=mr.ci,
            effect_size=mr.effect_size,
            n_per_group=mr.n_per_group,
        )

    def demographic_parity_difference(self, y_pred, sensitive):
        key = f"dp:{id(y_pred)}:{id(sensitive)}:{self.min_group_size}"
        if key in self._cache: #trivial cache for repeated calls
            return self._cache[key]
        
        mr = self._adapter.demographic_parity_difference(
            y_true=None,
            y_pred=np.asarray(y_pred),
            sensitive=np.asarray(sensitive),
            min_group_size=self.min_group_size,
        )
        res = self._to_result(mr)
        self._cache[key] = res
        return res
        # y_pred = np.asarray(y_pred)
        # sensitive = np.asarray(sensitive)
        # groups = np.unique(sensitive)
        # rates, n_per = [], {}
        # for g in groups:
        #     m = sensitive == g
        #     n = m.sum()
        #     if n >= self.min_group_size:
        #         rates.append(y_pred[m].mean())
        #         n_per[str(g)] = int(n)
        # value = np.nan if len(rates) < 2 else float(np.max(rates) - np.min(rates))
        # return Result(metric="demographic_parity_difference", value=value, n_per_group=n_per)
    def equalized_odds_difference(self, y_true, y_pred, sensitive):
        key = f"eod:{id(y_true)}:{id(y_pred)}:{id(sensitive)}:{self.min_group_size}"
        if key in self._cache: #trivial cache for repeated calls
            return self._cache[key]
        
        mr = self._adapter.equalized_odds_difference(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            sensitive=np.asarray(sensitive),
            min_group_size=self.min_group_size,
        )
        res = self._to_result(mr)
        self._cache[key] = res
        return res
    
    @property
    def backend(self) -> str:
        """Name of the currently selected backend."""
        return self._backend
    