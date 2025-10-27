from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from ..utils.validation import coerce_arrays, check_lengths
from .base import MetricResult
from .native_adapter import NativeAdapter
from .fairlearn_adapter import FairlearnAdapter
from .aequitas_adapter import AequitasAdapter
from ..utils.intersectional import build_intersectional_labels, min_group_mask, group_sizes

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

    Notes on Metic Engine Design:
    - add mae_parity_difference (regression).
    - expose intersectionality controls via method kwargs.
    - keep a tiny call-scoped cache for speed (e.g., repeated calls with same inputs).
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

    def demographic_parity_difference(
            self, 
            y_pred, 
            sensitive, 
            *, 
            intersectional: bool = False, 
            attrs_df: Optional[pd.DataFrame] = None, 
            attr_columns: Optional[List[str]] = None, 
            columns: Optional[List[str]] = None
        ):
        """
        Maximum difference in positive prediction rates across groups.

        If intersectional is True, supply attrs_df (and columns if subset) to construct mukti-attribute group labels.

        """
        # key = f"dp:{id(y_pred)}:{id(sensitive)}:{self.min_group_size}"
        # if key in self._cache: #trivial cache for repeated calls
        #     return self._cache[key]
        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df must be provided when intersectional=True.")
            labels = build_intersectional_labels(
                attrs_df=attrs_df,
                columns=columns,
                include_na=(self.nan_policy == "include")
            )
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                raise Result("demographic_parity_difference", np.nan, n_per_group={})
            n_per = group_sizes(labels[mask])
            #delegate to adapter using intersectional labels as 'sensitive'
            mr = self._adapter.demographic_parity_difference(
                y_true=None,
                y_pred=np.asarray(y_pred)[mask],
                sensitive=np.asarray(labels[mask]),
                min_group_size=self.min_group_size,
            )
            # prefer adapter's n_per_group if available; otherwise use our computed one
            if mr.n_per_group is None:
                mr.n_per_group = n_per
            return self._to_result(mr)
        
        #Non-intersectional path
        mr = self._adapter.demographic_parity_difference(
            y_true=None,
            y_pred=np.asarray(y_pred),
            sensitive=np.asarray(sensitive),
            min_group_size=self.min_group_size,
        )
        res = self._to_result(mr)
        # self._cache[key] = res
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

    # def equalized_odds_difference(self, y_true, y_pred, sensitive):
    #     key = f"eod:{id(y_true)}:{id(y_pred)}:{id(sensitive)}:{self.min_group_size}"
    #     if key in self._cache: #trivial cache for repeated calls
    #         return self._cache[key]
        
    #     mr = self._adapter.equalized_odds_difference(
    #         y_true=np.asarray(y_true),
    #         y_pred=np.asarray(y_pred),
    #         sensitive=np.asarray(sensitive),
    #         min_group_size=self.min_group_size,
    #     )
    #     res = self._to_result(mr)
    #     self._cache[key] = res
    #     return res

    def equalized_odds_difference(self, y_true, y_pred, sensitive, 
                                 *, 
                                 intersectional: bool = False, 
                                 attrs_df: Optional[pd.DataFrame] = None,  
                                 columns: Optional[List[str]] = None):
        """
        Maximum difference in TPR and FPR across groups.
        """
        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df must be provided when intersectional=True.")
            labels = build_intersectional_labels(
                attrs_df=attrs_df,
                columns=columns,
                include_na=(self.nan_policy == "include")
            )
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                raise Result("equalized_odds_difference", np.nan, n_per_group={})
            n_per = group_sizes(labels[mask])
            #delegate to adapter using intersectional labels as 'sensitive'
            mr = self._adapter.equalized_odds_difference(
                y_true=np.asarray(y_true)[mask],
                y_pred=np.asarray(y_pred)[mask],
                sensitive=np.asarray(labels[mask]),
                min_group_size=self.min_group_size,
            )
            # prefer adapter's n_per_group if available; otherwise use our computed one
            if mr.n_per_group is None:
                mr.n_per_group = n_per
            return self._to_result(mr)
        #Non-intersectional path
        mr = self._adapter.equalized_odds_difference(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            sensitive=np.asarray(sensitive),
            min_group_size=self.min_group_size,
        )
        return self._to_result(mr)

    # -----------------------------
    # Regression metrics
    # -----------------------------
    def mae_parity_difference(self, y_true, y_pred, sensitive, *, intersectional: bool = False, 
                             attrs_df: Optional[pd.DataFrame] = None,  
                             columns: Optional[List[str]] = None):
        """
        Mean Absolute Error difference across groups.
        max_group(MAE) - min_group(MAE)

        This is implemented natively here (adapters are for classification-focused libraries).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if intersectional:
            if attrs_df is None:
                raise ValueError("attrs_df must be provided when intersectional=True.")
            labels = build_intersectional_labels(
                attrs_df=attrs_df,
                columns=columns,
                include_na=(self.nan_policy == "include")
            )
            mask = min_group_mask(labels, self.min_group_size)
            if mask.sum() == 0:
                raise Result("mae_parity_difference", np.nan, n_per_group={})
            labels = np.asarray(labels)
            labels = labels[mask]; y_true = y_true[mask]; y_pred = y_pred[mask]
            groups = np.unique(labels)
            maes, n_per = {}, {}
            for g in groups:
                m = (labels == g)
                n = int(m.sum())
                if n >= self.min_group_size:
                    mae = np.mean(np.abs(y_true[m] - y_pred[m]))
                    maes[str(g)] = float(mae)
                    n_per[str(g)] = n
            if len(maes) < 2:
                return Result("mae_parity_difference", np.nan, n_per_group=n_per)
            diff = max(maes.values()) - min(maes.values())
            return Result("mae_parity_difference", float(diff), n_per_group=n_per)
        
        # Non-intersectional path
        s = np.asarray(sensitive)
        groups = np.unique(s)
        maes, n_per = {}, {}
        for g in groups:
            m = (s == g)
            n = int(m.sum())
            if n >= self.min_group_size:
                mae = np.mean(np.abs(y_true[m] - y_pred[m]))
                maes[str(g)] = float(mae)
                n_per[str(g)] = n
        if len(maes) < 2:
            return Result("mae_parity_difference", np.nan, n_per_group=n_per)
        diff = max(maes.values()) - min(maes.values())
        return Result("mae_parity_difference", float(diff), n_per_group=n_per)
    
    @property
    def backend(self) -> str:
        """Name of the currently selected backend."""
        return self._backend
    