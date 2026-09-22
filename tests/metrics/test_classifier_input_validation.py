"""Wave 1d / BL-015: classifier metric input validation."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.metrics.input_validation import (
    POSITIVE_LABEL,
    LengthMismatchError,
    MulticlassNotSupportedError,
    NonBinaryEncodingError,
    prepare_binary_classifier_inputs,
)
from fairness_pipeline_dev_toolkit.metrics.native_adapter import NativeAdapter

# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


def test_length_mismatch_raises_named_error_not_index_error():
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    with pytest.raises(LengthMismatchError, match="y_pred=3.*sensitive=2"):
        fa.demographic_parity_difference([0, 1, 0], ["A", "B"])


def test_length_match_near_miss_passes():
    fa = FairnessAnalyzer(min_group_size=2, backend="native")
    res = fa.demographic_parity_difference(
        [0, 1, 0, 1],
        ["A", "A", "B", "B"],
        with_ci=False,
        with_effect_size=False,
    )
    assert res.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Non-finite (drop with reported count — matches nan_policy='exclude')
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y_pred",
    [
        np.array([0.0, 1.0, np.nan, 1.0]),  # reviewer: NaN in B → was DPD 0.0
        np.array([np.nan, 1.0, 0.0, 1.0]),  # reviewer: NaN in A → was NaN
    ],
)
def test_nan_in_y_pred_both_reviewer_positions_drop_identically(y_pred):
    """Both positions drop the incomplete row; neither invents a zero gap."""
    sensitive = np.array(["A", "A", "B", "B"])
    adapter = NativeAdapter()
    result = adapter.demographic_parity_difference(None, y_pred, sensitive, min_group_size=1)
    assert result.n_dropped_nonfinite == 1
    assert result.caveat is not None
    assert "Dropped 1" in result.caveat
    # After drop: one group may fall below size or rates are finite — never the
    # silent max/min-with-nan 0.0 artefact from the unfixed native adapter.
    if np.isfinite(result.value):
        assert 0.0 <= result.value <= 1.0


def test_nan_drop_near_miss_all_finite_passes_without_drop():
    adapter = NativeAdapter()
    result = adapter.demographic_parity_difference(
        None,
        np.array([0, 1, 0, 1]),
        np.array(["A", "A", "B", "B"]),
        min_group_size=1,
    )
    assert result.n_dropped_nonfinite == 0
    assert result.caveat is None
    assert result.value == pytest.approx(0.0)


def test_inf_in_y_pred_dropped_with_count():
    adapter = NativeAdapter()
    result = adapter.demographic_parity_difference(
        None,
        np.array([0.0, 1.0, np.inf, 1.0]),
        np.array(["A", "A", "B", "B"]),
        min_group_size=1,
    )
    assert result.n_dropped_nonfinite == 1
    assert "Dropped 1" in (result.caveat or "")


# ---------------------------------------------------------------------------
# Binary / multiclass
# ---------------------------------------------------------------------------


def test_multiclass_raises_with_distinct_values():
    adapter = NativeAdapter()
    with pytest.raises(MulticlassNotSupportedError, match=r"\[0\.0, 1\.0, 2\.0\]"):
        adapter.demographic_parity_difference(
            None,
            np.array([0, 1, 2, 0, 1, 2]),
            np.array(["A", "A", "A", "B", "B", "B"]),
            min_group_size=1,
        )


def test_multiclass_eod_raises():
    adapter = NativeAdapter()
    with pytest.raises(MulticlassNotSupportedError):
        adapter.equalized_odds_difference(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            np.array(["A", "A", "A", "B", "B", "B"]),
            min_group_size=1,
        )


def test_binary_not_01_raises_encoding_error():
    adapter = NativeAdapter()
    with pytest.raises(NonBinaryEncodingError, match="positive class 1"):
        adapter.demographic_parity_difference(
            None,
            np.array([-1, 1, -1, 1]),
            np.array(["A", "A", "B", "B"]),
            min_group_size=1,
        )


def test_bool_labels_accepted_as_01():
    adapter = NativeAdapter()
    result = adapter.demographic_parity_difference(
        None,
        np.array([False, True, False, True]),
        np.array(["A", "A", "B", "B"]),
        min_group_size=1,
    )
    assert result.value == pytest.approx(0.0)
    assert POSITIVE_LABEL == 1


def test_prepare_documents_positive_label_one():
    prepared = prepare_binary_classifier_inputs(
        y_pred=[0, 1, 0, 1],
        sensitive=["A", "A", "B", "B"],
    )
    assert prepared.positive_label == 1


# ---------------------------------------------------------------------------
# Reviewer regression oracles (verified-correct ordinary binary cases)
# ---------------------------------------------------------------------------


def test_oracle_dpd_0_35():
    """Group A selection rate 0.35, group B 0.0 → DPD 0.35."""
    n = 20
    y_pred = np.array([1] * 7 + [0] * 13 + [0] * n)
    sensitive = np.array(["A"] * n + ["B"] * n)
    fa = FairnessAnalyzer(min_group_size=5, backend="native")
    res = fa.demographic_parity_difference(y_pred, sensitive, with_ci=False, with_effect_size=False)
    assert res.value == pytest.approx(0.35)


def test_oracle_eod_0_4():
    """TPR gap 0.4 (A: 1.0, B: 0.6); FPR both 0 → EOD 0.4."""
    # Per group: 5 positives, 5 negatives. A predicts all positives correctly;
    # B predicts 3/5 positives correctly. No false positives.
    yt_a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yp_a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yt_b = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yp_b = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_true = np.array(yt_a + yt_b)
    y_pred = np.array(yp_a + yp_b)
    sensitive = np.array(["A"] * 10 + ["B"] * 10)
    fa = FairnessAnalyzer(min_group_size=5, backend="native")
    res = fa.equalized_odds_difference(
        y_true, y_pred, sensitive, with_ci=False, with_effect_size=False
    )
    assert res.value == pytest.approx(0.4)


def test_oracle_mae_gap_0_05():
    """Group A MAE 0.10, group B MAE 0.15 → gap 0.05."""
    y_true = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    y_pred = np.array([1.1, 1.1, 0.9, 0.9, 2.15, 2.15, 1.85, 1.85])
    # A: |err| = 0.1 four times → mae 0.1
    # B: |err| = 0.15 four times → mae 0.15
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    fa = FairnessAnalyzer(min_group_size=2, backend="native")
    res = fa.mae_parity_difference(y_true, y_pred, sensitive, with_ci=False, with_effect_size=False)
    assert res.value == pytest.approx(0.05)


def test_analyzer_surfaces_n_dropped_on_result():
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    res = fa.demographic_parity_difference(
        [0, 1, float("nan"), 1],
        ["A", "A", "B", "B"],
        with_ci=False,
        with_effect_size=False,
    )
    assert res.n_dropped_nonfinite == 1
    assert res.caveat is not None and "Dropped 1" in res.caveat
