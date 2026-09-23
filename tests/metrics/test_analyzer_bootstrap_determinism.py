"""Regression tests for analyzer bootstrap statistic determinism (Wave 1a / BL-013)."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.metrics.core import (
    FairnessAnalyzer,
    _sens_keys,
    dpd_stat_from_indices,
    eod_stat_from_indices,
    mae_gap_stat_from_indices,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci


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


class TestDPDBootstrapOracle:
    def test_dpd_ci_matches_independent_index_bootstrap(self):
        """Numerical oracle: analyzer CI equals an independent same-seed bootstrap."""
        rng = np.random.default_rng(0)
        n_per = 40
        y_pred = rng.integers(0, 2, size=n_per * 2).astype(float)
        sensitive = np.repeat([0, 1], n_per)

        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        res = fa.demographic_parity_difference(
            y_pred,
            sensitive,
            with_ci=True,
            ci_level=0.95,
            ci_method="percentile",
            ci_samples=500,
            with_effect_size=False,
        )
        assert res.ci is not None

        group_of = _sens_keys(sensitive)
        groups = ["0", "1"]
        obs_idx = np.arange(len(y_pred), dtype=int)

        def stat_fn(sample_idx):
            return dpd_stat_from_indices(sample_idx, y_pred, group_of, groups)

        # bootstrap_ci default random_state=42 matches analyzer (no seed override).
        expected = bootstrap_ci(
            obs_idx, stat_fn, B=500, level=0.95, method="percentile", random_state=42
        )
        assert res.ci[0] == pytest.approx(expected[0], abs=1e-12)
        assert res.ci[1] == pytest.approx(expected[1], abs=1e-12)
        assert res.ci[0] < res.ci[1]
        assert np.isfinite(res.ci[0]) and np.isfinite(res.ci[1])


@pytest.mark.calibration
def test_dpd_percentile_ci_undercovers_at_equality():
    """Documents remaining BL-014: at true DPD=0, nominal 95% percentile CIs undercover.

    Wave 1a fixed statistic determinism; this simulation still sees ~0 coverage,
    so the boundary/max−min explanation stands. Locked so a silent 'fix' that
    only restores the old non-deterministic draw cannot claim calibration recovery.
    """
    n_sim = 40
    n_per = 100
    n_groups = 3
    b = 200
    covers = 0
    for seed in range(n_sim):
        rng = np.random.default_rng(seed)
        y_pred = rng.integers(0, 2, size=n_per * n_groups).astype(float)
        sensitive = np.repeat(np.arange(n_groups), n_per)
        fa = FairnessAnalyzer(min_group_size=30, backend="native")
        res = fa.demographic_parity_difference(
            y_pred,
            sensitive,
            with_ci=True,
            ci_level=0.95,
            ci_method="percentile",
            ci_samples=b,
            with_effect_size=False,
        )
        lo, hi = res.ci
        if lo <= 0.0 <= hi:
            covers += 1
    rate = covers / n_sim
    # Pre- and post-determinism-fix coverage is ~0 under this protocol.
    assert rate < 0.15, f"unexpected coverage {rate:.2%} — revisit BL-014 assumptions"
