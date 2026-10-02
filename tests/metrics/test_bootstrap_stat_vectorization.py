"""Equivalence tests for the vectorized bootstrap statistic functions (perf pass).

``dpd_stat_from_indices`` / ``eod_stat_from_indices`` in
``fairness_pipeline_dev_toolkit.metrics.core`` were rewritten to replace a
Python loop over groups (each doing an O(n) boolean mask) with a couple of
vectorized ``np.bincount`` passes — a big win when the number of groups is
large (e.g. intersectional attributes). Since classifier metrics only ever
see binary {0, 1} labels, per-group sums are exact integers regardless of
accumulation order, so the new implementation must be *bit-identical* to the
original loop-based one, not just close. These tests pin that by keeping a
verbatim copy of the pre-optimization implementation and comparing outputs
across edge cases (empty group, single-row group, a NaN/"nan" group label,
and an intersectional-style many-group case).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.metrics.core import (
    _group_codes,
    _sens_keys,
    dpd_stat_from_indices,
    eod_stat_from_indices,
)


# ---- verbatim copies of the pre-optimization (loop-based) implementations ----


def _dpd_stat_ref(
    sample_idx: np.ndarray,
    y_pred: np.ndarray,
    group_of: np.ndarray,
    groups: Sequence[str],
) -> float:
    idxs = np.asarray(sample_idx, dtype=int)
    rates: List[float] = []
    for g in groups:
        sel = idxs[group_of[idxs] == g]
        if sel.size == 0:
            return float("nan")
        rates.append(float(y_pred[sel].mean()))
    return float(max(rates) - min(rates))


def _eod_stat_ref(
    sample_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_of: np.ndarray,
    groups: Sequence[str],
) -> float:
    idxs = np.asarray(sample_idx, dtype=int)
    tprs: List[float] = []
    fprs: List[float] = []
    for g in groups:
        sel = idxs[group_of[idxs] == g]
        if sel.size == 0:
            return float("nan")
        yt_g = y_true[sel]
        yp_g = y_pred[sel]
        pos = yt_g == 1
        neg = yt_g == 0
        tpr_g = np.nan if not np.any(pos) else float((yp_g[pos] == 1).mean())
        fpr_g = np.nan if not np.any(neg) else float((yp_g[neg] == 1).mean())
        if np.isfinite(tpr_g):
            tprs.append(tpr_g)
        if np.isfinite(fpr_g):
            fprs.append(fpr_g)
    tpr_gap = np.nan if len(tprs) < 2 else (max(tprs) - min(tprs))
    fpr_gap = np.nan if len(fprs) < 2 else (max(fprs) - min(fprs))
    if not np.isfinite(tpr_gap) and not np.isfinite(fpr_gap):
        return float("nan")
    return float(np.nanmax([tpr_gap, fpr_gap]))


def _assert_same(old: float, new: float) -> None:
    if np.isnan(old) or np.isnan(new):
        assert np.isnan(old) and np.isnan(new)
    else:
        assert np.allclose(old, new, rtol=0, atol=1e-12)


def _many_samples(n: int, rng: np.random.Generator, k: int = 40) -> List[np.ndarray]:
    """A handful of bootstrap-style resamples (with replacement) of size n."""
    return [rng.integers(0, n, n) for _ in range(k)]


class TestDPDEquivalence:
    def test_basic_random_case(self):
        rng = np.random.default_rng(0)
        n = 500
        y_pred = rng.integers(0, 2, size=n)
        sensitive = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        group_of = _sens_keys(sensitive)
        groups = ["A", "B", "C"]
        codes = _group_codes(group_of, groups)
        for sample in _many_samples(n, rng):
            old = _dpd_stat_ref(sample, y_pred, group_of, groups)
            new = dpd_stat_from_indices(sample, y_pred, group_of, groups)
            new_cached = dpd_stat_from_indices(sample, y_pred, group_of, groups, codes=codes)
            _assert_same(old, new)
            _assert_same(old, new_cached)

    def test_empty_group_in_resample(self):
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        group_of = _sens_keys(np.array([0, 0, 0, 1, 1, 1]))
        groups = ["0", "1"]
        # Only group "0" rows selected -> group "1" absent from the resample.
        sample = np.array([0, 1, 2, 0, 1, 2])
        old = _dpd_stat_ref(sample, y_pred, group_of, groups)
        new = dpd_stat_from_indices(sample, y_pred, group_of, groups)
        assert np.isnan(old) and np.isnan(new)

    def test_single_row_group(self):
        y_pred = np.array([1, 0, 1, 0, 1])
        group_of = _sens_keys(np.array([0, 0, 0, 0, 1]))
        groups = ["0", "1"]
        # Resample hits the single "1" row plus repeats of "0" rows.
        sample = np.array([0, 1, 2, 3, 4, 4, 4])
        old = _dpd_stat_ref(sample, y_pred, group_of, groups)
        new = dpd_stat_from_indices(sample, y_pred, group_of, groups)
        _assert_same(old, new)

    def test_nan_group_label_excluded(self):
        # A "nan" string group label present in group_of but not in `groups`
        # (as happens after min-group-size / nan_policy="exclude" filtering)
        # must simply be ignored by both implementations.
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1])
        group_of = np.array(["0", "0", "0", "1", "1", "1", "nan"])
        groups = ["0", "1"]
        sample = np.array([0, 1, 2, 3, 4, 5, 6, 6, 6])
        old = _dpd_stat_ref(sample, y_pred, group_of, groups)
        new = dpd_stat_from_indices(sample, y_pred, group_of, groups)
        _assert_same(old, new)

    def test_intersectional_many_groups(self):
        rng = np.random.default_rng(3)
        n = 2000
        y_pred = rng.integers(0, 2, size=n)
        race = rng.choice(["A", "B", "C", "D", "E"], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        gender = rng.choice(["M", "F"], size=n)
        age = rng.choice(["<30", "30-50", "50+"], size=n)
        labels = np.array([f"{r}_{g}_{a}" for r, g, a in zip(race, gender, age)])
        group_of = _sens_keys(labels)
        groups = sorted(set(labels.tolist()))
        codes = _group_codes(group_of, groups)
        for sample in _many_samples(n, rng, k=20):
            old = _dpd_stat_ref(sample, y_pred, group_of, groups)
            new = dpd_stat_from_indices(sample, y_pred, group_of, groups, codes=codes)
            _assert_same(old, new)


class TestEODEquivalence:
    def test_basic_random_case(self):
        rng = np.random.default_rng(1)
        n = 500
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)
        sensitive = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        group_of = _sens_keys(sensitive)
        groups = ["A", "B", "C"]
        codes = _group_codes(group_of, groups)
        for sample in _many_samples(n, rng):
            old = _eod_stat_ref(sample, y_true, y_pred, group_of, groups)
            new = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups)
            new_cached = eod_stat_from_indices(
                sample, y_true, y_pred, group_of, groups, codes=codes
            )
            _assert_same(old, new)
            _assert_same(old, new_cached)

    def test_empty_group_in_resample(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        group_of = _sens_keys(np.array([0, 0, 0, 1, 1, 1]))
        groups = ["0", "1"]
        sample = np.array([0, 1, 2, 0, 1, 2])
        old = _eod_stat_ref(sample, y_true, y_pred, group_of, groups)
        new = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups)
        assert np.isnan(old) and np.isnan(new)

    def test_single_row_group_all_one_label(self):
        # Group "1" has a single row and it is all y_true=1 -> FPR undefined for that group.
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        group_of = _sens_keys(np.array([0, 0, 0, 0, 1]))
        groups = ["0", "1"]
        sample = np.array([0, 1, 2, 3, 4, 4, 4])
        old = _eod_stat_ref(sample, y_true, y_pred, group_of, groups)
        new = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups)
        _assert_same(old, new)

    def test_nan_group_label_excluded(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 1, 0, 0])
        group_of = np.array(["0", "0", "0", "1", "1", "1", "nan"])
        groups = ["0", "1"]
        sample = np.array([0, 1, 2, 3, 4, 5, 6, 6, 6])
        old = _eod_stat_ref(sample, y_true, y_pred, group_of, groups)
        new = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups)
        _assert_same(old, new)

    def test_intersectional_many_groups(self):
        rng = np.random.default_rng(4)
        n = 2000
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)
        race = rng.choice(["A", "B", "C", "D", "E"], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        gender = rng.choice(["M", "F"], size=n)
        age = rng.choice(["<30", "30-50", "50+"], size=n)
        labels = np.array([f"{r}_{g}_{a}" for r, g, a in zip(race, gender, age)])
        group_of = _sens_keys(labels)
        groups = sorted(set(labels.tolist()))
        codes = _group_codes(group_of, groups)
        for sample in _many_samples(n, rng, k=20):
            old = _eod_stat_ref(sample, y_true, y_pred, group_of, groups)
            new = eod_stat_from_indices(sample, y_true, y_pred, group_of, groups, codes=codes)
            _assert_same(old, new)


class TestFairnessAnalyzerIntersectionalCIUnchanged:
    """End-to-end: analyzer CI on intersectional data is unchanged by the vectorization."""

    def test_dpd_intersectional_ci_matches_reference_bootstrap(self):
        from fairness_pipeline_dev_toolkit.metrics.core import FairnessAnalyzer
        from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci

        rng = np.random.default_rng(5)
        n = 1500
        y_pred = rng.integers(0, 2, size=n)
        attrs = {
            "race": rng.choice(["A", "B", "C"], size=n),
            "gender": rng.choice(["M", "F"], size=n),
        }
        import pandas as pd

        attrs_df = pd.DataFrame(attrs)

        fa = FairnessAnalyzer(min_group_size=10, backend="native")
        res = fa.demographic_parity_difference(
            y_pred,
            sensitive=None,
            intersectional=True,
            attrs_df=attrs_df,
            with_ci=True,
            ci_samples=300,
            with_effect_size=False,
        )
        assert res.ci is not None

        # Reconstruct the same groups/labels the analyzer used internally and
        # verify the CI against the old (reference) loop-based statistic.
        from fairness_pipeline_dev_toolkit.utils.intersectional import (
            build_intersectional_labels,
            min_group_mask,
        )

        labels = build_intersectional_labels(attrs_df, columns=None, include_na=False)
        labels_array = np.asarray(labels, dtype=object)
        mask = np.asarray(min_group_mask(labels_array, 10), dtype=bool)
        sens = labels_array[mask]
        yp = np.asarray(y_pred)[mask]

        group_of = _sens_keys(sens)
        groups = [str(g) for g, n_g in zip(*np.unique(group_of, return_counts=True)) if n_g >= 10]
        obs_idx = np.arange(len(yp), dtype=int)

        def ref_stat(sample_idx):
            return _dpd_stat_ref(sample_idx, yp, group_of, groups)

        expected = bootstrap_ci(obs_idx, ref_stat, B=300, level=0.95, method="percentile")
        assert res.ci[0] == pytest.approx(expected[0], abs=1e-12)
        assert res.ci[1] == pytest.approx(expected[1], abs=1e-12)
