"""Shared adapter contract suite (BL-027 / Wave 4 PR B).

Parametrized over every installed adapter. Adapters must agree, or the
difference must be a documented, tested exception.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.metrics.aequitas_adapter import AequitasAdapter
from fairness_pipeline_dev_toolkit.metrics.fairlearn_adapter import FairlearnAdapter
from fairness_pipeline_dev_toolkit.metrics.native_adapter import NativeAdapter


def _available_adapters():
    candidates = [NativeAdapter(), FairlearnAdapter(), AequitasAdapter()]
    return [a for a in candidates if a.available()]


ADAPTERS = _available_adapters()
ADAPTER_IDS = [a.name for a in ADAPTERS]


@pytest.fixture(params=ADAPTERS, ids=ADAPTER_IDS)
def adapter(request):
    return request.param


def test_balanced_gap_zero(adapter):
    """Balanced selection / TPR / FPR → DPD and EOD are 0."""
    y_true = np.tile([1, 0], 20)
    y_pred = np.tile([1, 0], 20)
    sensitive = np.array(["A"] * 20 + ["B"] * 20)
    # Mirror labels in B so both groups look the same.
    y_true[20:] = y_true[:20]
    y_pred[20:] = y_pred[:20]

    dpd = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=5)
    eod = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=5)
    assert dpd.value == pytest.approx(0.0)
    assert eod.value == pytest.approx(0.0)


def test_known_gap_oracle(adapter):
    """Known selection-rate gap of 1.0 (A always predicts 1, B always 0)."""
    y_pred = np.array([1] * 20 + [0] * 20)
    y_true = np.array([1, 0] * 20)
    sensitive = np.array(["A"] * 20 + ["B"] * 20)
    dpd = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=5)
    assert dpd.value == pytest.approx(1.0)


def test_group_with_no_positives_eod_undefined(adapter):
    sensitive = np.repeat(["a", "b"], 40)
    y_true = np.tile([1.0, 0.0], 40)
    y_true[40:] = 0.0  # group b: no positives
    y_pred = np.tile([1.0, 0.0, 0.0, 1.0], 20)
    eod = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=10)
    assert math.isnan(eod.value)


def test_group_with_no_negatives_eod_undefined(adapter):
    sensitive = np.repeat(["a", "b"], 40)
    y_true = np.tile([1.0, 0.0], 40)
    y_true[40:] = 1.0  # group b: no negatives
    y_pred = np.tile([1.0, 0.0, 0.0, 1.0], 20)
    eod = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=10)
    assert math.isnan(eod.value)


def test_single_group_undefined(adapter):
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    sensitive = np.array(["only"] * 6)
    dpd = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)
    eod = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)
    assert math.isnan(dpd.value)
    assert math.isnan(eod.value)


def test_multiclass_rejected(adapter):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 0, 2])
    sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    with pytest.raises((ValueError, TypeError)):
        adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)


def test_nan_input_wave1_contract(adapter):
    """Non-finite y_pred rows are dropped with a caveat (Wave 1)."""
    y_pred = np.array([0.0, 1.0, np.nan, 1.0, 0.0, 1.0])
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    dpd = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)
    assert dpd.n_dropped_nonfinite is not None
    assert dpd.n_dropped_nonfinite >= 1
    assert dpd.caveat is not None


def test_string_categorical_sensitive(adapter):
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    sensitive = pd.Series(["woman", "woman", "woman", "woman", "man", "man", "man", "man"]).astype(
        "category"
    )
    dpd = adapter.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)
    eod = adapter.equalized_odds_difference(y_true, y_pred, sensitive, min_group_size=2)
    assert np.isfinite(dpd.value)
    assert np.isfinite(eod.value)
    assert set(dpd.n_per_group) == {"woman", "man"}


def test_adapters_agree_on_balanced_and_oracle():
    """Cross-adapter agreement on the shared finite cases."""
    adapters = _available_adapters()
    y_true = np.tile([1, 0], 20)
    y_pred = np.tile([1, 0], 20)
    sensitive = np.array(["A"] * 20 + ["B"] * 20)
    y_true[20:] = y_true[:20]
    y_pred[20:] = y_pred[:20]
    values = [
        a.demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=5).value
        for a in adapters
    ]
    assert all(v == pytest.approx(values[0]) for v in values)

    y_pred_gap = np.array([1] * 20 + [0] * 20)
    gaps = [
        a.demographic_parity_difference(y_true, y_pred_gap, sensitive, min_group_size=5).value
        for a in adapters
    ]
    assert all(v == pytest.approx(1.0) for v in gaps)
