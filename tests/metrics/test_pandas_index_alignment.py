"""Wave 1e / BL-020: pandas index alignment must not be silently positional."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer, IndexMismatchError
from fairness_pipeline_dev_toolkit.metrics.native_adapter import NativeAdapter


def test_reviewer_mismatched_index_order_raises():
    """Same data, different index order: was DPD 0 positional vs 1 aligned → now error."""
    y_pred = pd.Series([1, 0, 1, 0], index=[0, 2, 1, 3])
    sensitive = pd.Series(["A", "A", "B", "B"], index=[0, 1, 2, 3])
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    with pytest.raises(IndexMismatchError, match="unequal indices|reindex"):
        fa.demographic_parity_difference(y_pred, sensitive, with_ci=False, with_effect_size=False)


def test_matching_indices_unchanged_dpd_one():
    """Regression: matching Series indices still yield DPD 1 for the aligned case."""
    y_pred = pd.Series([1, 1, 0, 0], index=[0, 1, 2, 3])
    sensitive = pd.Series(["A", "A", "B", "B"], index=[0, 1, 2, 3])
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    res = fa.demographic_parity_difference(y_pred, sensitive, with_ci=False, with_effect_size=False)
    assert res.value == pytest.approx(1.0)


def test_mixed_series_and_array_uses_positional():
    """One Series + arrays: no second index to disagree with — positional OK."""
    y_pred = pd.Series([1, 1, 0, 0], index=[10, 20, 30, 40])
    sensitive = np.array(["A", "A", "B", "B"])
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    res = fa.demographic_parity_difference(y_pred, sensitive, with_ci=False, with_effect_size=False)
    assert res.value == pytest.approx(1.0)


def test_non_pandas_inputs_unaffected():
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    res = fa.demographic_parity_difference(
        [1, 1, 0, 0],
        ["A", "A", "B", "B"],
        with_ci=False,
        with_effect_size=False,
    )
    assert res.value == pytest.approx(1.0)


def test_adapter_direct_call_also_checks_indices():
    adapter = NativeAdapter()
    y_pred = pd.Series([1, 0, 1, 0], index=[0, 2, 1, 3])
    sensitive = pd.Series(["A", "A", "B", "B"], index=[0, 1, 2, 3])
    with pytest.raises(IndexMismatchError):
        adapter.demographic_parity_difference(None, y_pred, sensitive, min_group_size=1)


def test_wave1d_oracle_dpd_0_35_still_holds():
    n = 20
    y_pred = np.array([1] * 7 + [0] * 13 + [0] * n)
    sensitive = np.array(["A"] * n + ["B"] * n)
    fa = FairnessAnalyzer(min_group_size=5, backend="native")
    res = fa.demographic_parity_difference(y_pred, sensitive, with_ci=False, with_effect_size=False)
    assert res.value == pytest.approx(0.35)


def test_wave1d_oracle_eod_0_4_still_holds():
    yt_a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yp_a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yt_b = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yp_b = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    fa = FairnessAnalyzer(min_group_size=5, backend="native")
    res = fa.equalized_odds_difference(
        np.array(yt_a + yt_b),
        np.array(yp_a + yp_b),
        np.array(["A"] * 10 + ["B"] * 10),
        with_ci=False,
        with_effect_size=False,
    )
    assert res.value == pytest.approx(0.4)


def test_wave1d_oracle_mae_gap_0_05_still_holds():
    y_true = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    y_pred = np.array([1.1, 1.1, 0.9, 0.9, 2.15, 2.15, 1.85, 1.85])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    fa = FairnessAnalyzer(min_group_size=2, backend="native")
    res = fa.mae_parity_difference(y_true, y_pred, sensitive, with_ci=False, with_effect_size=False)
    assert res.value == pytest.approx(0.05)
