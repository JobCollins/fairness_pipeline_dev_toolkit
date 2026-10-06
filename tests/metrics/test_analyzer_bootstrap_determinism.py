"""Analyzer CI determinism, oracles and calibration (Wave 1a / BL-013, Wave 3a / BL-014, BL-031)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.metrics.core import (
    BCA_MIN_STRATUM_SIZE,
    FairnessAnalyzer,
    _sens_keys,
    dpd_stat_from_indices,
    eod_stat_from_indices,
    mae_gap_stat_from_indices,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import (
    _percentile_ci,
    stratified_bootstrap_replicates,
)
from fairness_pipeline_dev_toolkit.stats.gap_intervals import binary_gap_interval


class TestAnalyzerStatDeterminism:
    def test_dpd_stat_deterministic_in_sample(self):
        y_pred = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        group_of = _sens_keys(np.array([0, 0, 0, 1, 1, 1]))
        groups = ["0", "1"]
        sample = np.array([0, 2, 1, 3, 5, 4])
        a = dpd_stat_from_indices(sample, y_pred, group_of, groups)
        b = dpd_stat_from_indices(sample, y_pred, group_of, groups)
        assert a == b
        assert np.isfinite(a)

    def test_dpd_stat_empty_group_returns_nan(self):
        y_pred = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        group_of = _sens_keys(np.array([0, 0, 0, 1, 1, 1]))
        # Only group 0 indices — group 1 absent.
        sample = np.array([0, 1, 2, 0, 1, 2])
        assert np.isnan(dpd_stat_from_indices(sample, y_pred, group_of, ["0", "1"]))

    def test_eod_and_mae_stats_deterministic(self):
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=float)
        y_pred = np.array([1, 0, 0, 1, 1, 0], dtype=float)
        group_of = _sens_keys(np.array([0, 0, 0, 1, 1, 1]))
        groups = ["0", "1"]
        sample = np.array([0, 1, 2, 3, 4, 5])
        e1 = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups)
        e2 = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups)
        assert e1 == e2
        abs_err = np.abs(y_true - y_pred)
        m1 = mae_gap_stat_from_indices(sample, abs_err, group_of, groups)
        m2 = mae_gap_stat_from_indices(sample, abs_err, group_of, groups)
        assert m1 == m2


def _two_group_data(n_per: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_pred = rng.integers(0, 2, size=n_per * 2).astype(float)
    sensitive = np.repeat([0, 1], n_per)
    return y_pred, sensitive


class TestDPDOracles:
    def test_default_ci_is_analytic_simultaneous_interval(self):
        """Numerical oracle: default CI equals binary_gap_interval on the group counts."""
        y_pred, sensitive = _two_group_data()
        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        res = fa.demographic_parity_difference(y_pred, sensitive, with_effect_size=False)
        expected = binary_gap_interval(
            [y_pred[sensitive == 0].sum(), y_pred[sensitive == 1].sum()], [40, 40], 0.95
        )
        assert res.ci_kind == "simultaneous_pairwise"
        assert res.ci == pytest.approx(expected, abs=1e-15)
        assert res.ci[0] <= res.value <= res.ci[1]

    def test_percentile_opt_in_matches_stratified_bootstrap(self):
        """Numerical oracle: percentile opt-in == same-seed within-group bootstrap."""
        y_pred, sensitive = _two_group_data()
        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        res = fa.demographic_parity_difference(
            y_pred,
            sensitive,
            ci_method="percentile",
            ci_samples=500,
            with_effect_size=False,
            with_pvalue=False,
        )
        group_of = _sens_keys(sensitive)
        groups = ["0", "1"]
        strata = [np.flatnonzero(group_of == g) for g in groups]

        def stat_fn(sample_idx):
            return dpd_stat_from_indices(sample_idx, y_pred, group_of, groups)

        boot = stratified_bootstrap_replicates(strata, stat_fn, B=500, random_state=42)
        expected = _percentile_ci(boot, 0.95)
        assert res.ci_kind == "percentile"
        assert res.ci[0] == pytest.approx(expected[0], abs=1e-12)
        assert res.ci[1] == pytest.approx(expected[1], abs=1e-12)
        assert res.ci[0] < res.ci[1]

    def test_pvalue_reproducible_and_follows_with_ci(self):
        y_pred, sensitive = _two_group_data()
        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        a = fa.demographic_parity_difference(y_pred, sensitive)
        b = fa.demographic_parity_difference(y_pred, sensitive)
        assert a.p_value == b.p_value and 0 < a.p_value <= 1
        off = fa.demographic_parity_difference(y_pred, sensitive, with_ci=False)
        assert off.p_value is None and off.ci is None and off.ci_note is None
        only_p = fa.demographic_parity_difference(
            y_pred, sensitive, with_ci=False, with_pvalue=True
        )
        assert only_p.ci is None and only_p.p_value == a.p_value


def _equal_rate_coverage(ci_method: str, n_sim: int, **kwargs) -> float:
    covers = 0
    fa = FairnessAnalyzer(min_group_size=30, backend="native")
    sensitive = np.repeat(np.arange(3), 100)
    for seed in range(n_sim):
        rng = np.random.default_rng(seed)
        y_pred = rng.integers(0, 2, size=300).astype(float)
        res = fa.demographic_parity_difference(
            y_pred,
            sensitive,
            ci_method=ci_method,
            with_effect_size=False,
            with_pvalue=False,
            **kwargs,
        )
        lo, hi = res.ci
        covers += lo <= 0.0 <= hi
    return covers / n_sim


@pytest.mark.calibration
def test_dpd_default_ci_covers_zero_at_equality():
    """BL-014 fixed: the simultaneous default covers a true DPD of 0.

    3 groups x 100 rows, all rates 0.5. Wave 3a measured 0.970 coverage in this
    cell (S=500). With 200 runs the Monte Carlo SE is ~0.012, so the 0.90 floor is
    ~6 SE below the expected rate: a real regression to the old ~0 coverage
    fails, sampling noise does not.
    """
    rate = _equal_rate_coverage("simultaneous", n_sim=200)
    assert rate >= 0.90, f"coverage of 0 at equality {rate:.3f} < 0.90"


@pytest.mark.calibration
def test_dpd_percentile_opt_in_still_undercovers_at_equality():
    """Documents why percentile is opt-in only: stratified percentile CIs miss 0.

    Same design as above; Wave 3a measured 0.000 coverage (S=500).
    """
    rate = _equal_rate_coverage("percentile", n_sim=40, ci_samples=200)
    assert rate < 0.15, f"unexpected coverage {rate:.2%} — revisit BL-014 assumptions"


class TestBCaRefusePolicy:
    @pytest.mark.parametrize("n_minor", [1, 2, 3, 5])
    def test_bl031_minority_group_gives_undefined_ci_with_reason(self, n_minor):
        """BL-031 reproduction: majority 300, minority 1–5, min_group_size=1, B=1000."""
        rng = np.random.default_rng(31)
        sensitive = np.array(["maj"] * 300 + ["min"] * n_minor)
        y_pred = rng.binomial(1, 0.5, size=300 + n_minor).astype(float)
        fa = FairnessAnalyzer(min_group_size=1, backend="native")
        with pytest.warns(FutureWarning, match="bca"):
            res = fa.demographic_parity_difference(
                y_pred, sensitive, ci_method="bca", ci_samples=1000, with_pvalue=False
            )
        assert np.isfinite(res.value)
        assert res.ci is None
        assert res.ci_kind is None
        assert res.ci_note.startswith("undefined:")

    def test_bca_on_adequate_groups_returns_finite_ci(self):
        rng = np.random.default_rng(1)
        sensitive = np.repeat(["a", "b"], 60)
        y_pred = rng.binomial(1, 0.4, size=120).astype(float)
        fa = FairnessAnalyzer(min_group_size=30, backend="native")
        with pytest.warns(FutureWarning):
            res = fa.demographic_parity_difference(
                y_pred, sensitive, ci_method="bca", ci_samples=300, with_pvalue=False
            )
        assert res.ci_kind == "bca"
        assert np.all(np.isfinite(res.ci))

    def test_bca_floor_constant(self):
        assert BCA_MIN_STRATUM_SIZE == 10


def test_eod_stratified_replicates_never_drop_a_group():
    """Group × y_true stratification: every replicate keeps each group's positives
    and negatives, so the EOD statistic never silently drops a group's TPR/FPR."""
    rng = np.random.default_rng(3)
    sensitive = np.repeat(["a", "b", "c"], 40)
    y_true = rng.binomial(1, 0.5, size=120).astype(float)
    y_true[np.flatnonzero(sensitive == "c")] = 0.0
    y_true[np.flatnonzero(sensitive == "c")[0]] = 1.0  # group c: a single positive
    y_pred = rng.binomial(1, 0.5, size=120).astype(float)
    group_of = _sens_keys(sensitive)
    groups = ["a", "b", "c"]
    strata = [
        np.flatnonzero((group_of == g) & (y_true == lab)) for g in groups for lab in (1.0, 0.0)
    ]
    seen = []

    def stat_fn(idx):
        cells = {(group_of[i], y_true[i]) for i in idx}
        seen.append(len(cells))
        return eod_stat_from_indices(idx, y_true, y_pred, group_of, groups)

    boot = stratified_bootstrap_replicates(strata, stat_fn, B=500, random_state=0)
    assert np.all(np.isfinite(boot))
    assert set(seen) == {6}


def test_eod_group_without_positives_gives_undefined_ci():
    sensitive = np.repeat(["a", "b"], 40)
    y_true = np.tile([1.0, 0.0], 40)
    y_true[40:] = 0.0
    y_pred = np.tile([1.0, 0.0, 0.0, 1.0], 20)
    fa = FairnessAnalyzer(min_group_size=10, backend="native")
    res = fa.equalized_odds_difference(y_true, y_pred, sensitive)
    assert res.ci is None
    assert res.ci_note.startswith("undefined:empty_label_stratum")
    assert res.p_value is None


def test_too_few_groups_gives_undefined_note():
    fa = FairnessAnalyzer(min_group_size=30, backend="native")
    res = fa.demographic_parity_difference(
        np.ones(40), np.array(["a"] * 35 + ["b"] * 5), with_effect_size=False
    )
    assert res.ci is None
    assert res.ci_note.startswith("undefined:too_few_groups")


class TestMAEGap:
    """MAE gap: no interval passed decision 8 (issue #61), the p-value did."""

    def _data(self):
        rng = np.random.default_rng(3)
        sensitive = np.array(["a"] * 60 + ["b"] * 80)
        y_true = rng.normal(size=140)
        y_pred = y_true + rng.normal(scale=np.where(sensitive == "a", 1.0, 1.3))
        return y_true, y_pred, sensitive

    def test_default_ci_undefined_with_reason_and_pvalue(self):
        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        res = fa.mae_parity_difference(*self._data())
        assert res.ci is None and res.ci_kind is None
        assert res.ci_note.startswith("undefined:no_calibrated_interval")
        assert "issues/61" in res.ci_note
        assert res.p_value is not None and 0 < res.p_value <= 1
        assert np.isfinite(res.value)

    def test_percentile_opt_in_still_available(self):
        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        res = fa.mae_parity_difference(
            *self._data(), ci_method="percentile", ci_samples=200, with_pvalue=False
        )
        assert res.ci_kind == "percentile" and res.ci[0] <= res.ci[1]
        assert res.p_value is None


def test_unknown_ci_method_raises():
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    with pytest.raises(ValueError, match="ci_method"):
        fa.demographic_parity_difference([0, 1, 0, 1], ["a", "a", "b", "b"], ci_method="wald")


def test_simultaneous_ignores_ci_samples():
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fa.demographic_parity_difference(
            [0, 1, 0, 1, 1, 1], ["a"] * 3 + ["b"] * 3, ci_samples=0, with_pvalue=False
        )
    assert res.ci_kind == "simultaneous_pairwise"
    with pytest.raises(ValueError, match="ci_samples"):
        fa.demographic_parity_difference(
            [0, 1, 0, 1, 1, 1], ["a"] * 3 + ["b"] * 3, ci_method="percentile", ci_samples=0
        )
