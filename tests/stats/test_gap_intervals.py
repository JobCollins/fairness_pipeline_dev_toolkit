"""Unit tests for simultaneous gap intervals and the permutation gap test (Wave 3a / BL-014)."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.stats.gap_intervals import (
    binary_gap_interval,
    equalized_odds_gap_interval,
    permutation_gap_pvalue,
    simultaneous_gap_bounds,
    welch_gap_interval,
)


class TestSimultaneousGapBounds:
    def test_inversion_formula(self):
        assert simultaneous_gap_bounds([-0.05, 0.02], [0.05, 0.12]) == (0.0 + 0.02, 0.12)

    def test_negative_orientation_uses_minus_upper(self):
        lo, hi = simultaneous_gap_bounds([-0.3], [-0.1])
        assert lo == pytest.approx(0.1) and hi == pytest.approx(0.3)

    def test_clipped_to_max_gap(self):
        assert simultaneous_gap_bounds([0.9], [1.4]) == (0.9, 1.0)
        assert simultaneous_gap_bounds([0.9], [1.4], max_gap=None) == (0.9, 1.4)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            simultaneous_gap_bounds([0.2], [0.1])
        with pytest.raises(ValueError):
            simultaneous_gap_bounds([], [])
        with pytest.raises(IntervalUndefinedError):
            simultaneous_gap_bounds([np.nan], [0.1])


class TestBinaryGapInterval:
    def test_equal_groups_lower_bound_zero(self):
        lo, hi = binary_gap_interval([50, 50, 50], [100, 100, 100])
        assert lo == 0.0 and 0 < hi < 1

    def test_contains_observed_gap_and_is_deterministic(self):
        x, n = [30, 45, 60], [100, 120, 140]
        rates = np.array(x) / np.array(n)
        gap = rates.max() - rates.min()
        a = binary_gap_interval(x, n)
        assert a == binary_gap_interval(x, n)
        assert a[0] <= gap <= a[1]

    def test_wider_at_higher_level(self):
        lo90, hi90 = binary_gap_interval([20, 40], [100, 100], 0.90)
        lo99, hi99 = binary_gap_interval([20, 40], [100, 100], 0.99)
        assert lo99 <= lo90 and hi99 >= hi90

    def test_extreme_rates_stay_in_unit_interval(self):
        lo, hi = binary_gap_interval([0, 30], [30, 30])
        assert 0 < lo <= hi <= 1.0

    def test_contains_observed_gap_at_extreme_small_groups(self):
        """Without the hull, Agresti–Caffo shrinkage gave U=0.978 < 1.0 for 0/10 vs 10/10
        at level 0.8 (6586 of 92M exhaustive K=2 configurations, all at level 0.8)."""
        assert binary_gap_interval([0, 10], [10, 10], level=0.8)[1] == 1.0
        for n1 in range(1, 16):
            for n2 in (1, 5, 15, 100):
                for x1 in range(n1 + 1):
                    for x2 in (0, n2 // 2, n2):
                        for level in (0.8, 0.95):
                            lo, hi = binary_gap_interval([x1, x2], [n1, n2], level)
                            assert lo <= abs(x1 / n1 - x2 / n2) <= hi

    def test_edge_cases(self):
        with pytest.raises(IntervalUndefinedError) as e:
            binary_gap_interval([1], [10])
        assert e.value.reason == "too_few_groups"
        with pytest.raises(IntervalUndefinedError):
            binary_gap_interval([0, 1], [0, 10])
        with pytest.raises(ValueError):
            binary_gap_interval([11, 1], [10, 10])
        with pytest.raises(ValueError):
            binary_gap_interval([1, 1], [10, 10], level=1.0)


class TestEqualizedOddsGapInterval:
    def test_equal_groups(self):
        lo, hi = equalized_odds_gap_interval([40, 40], [50, 50], [10, 10], [50, 50])
        assert lo == 0.0 and hi > 0

    def test_covers_max_of_families(self):
        tp, pos, fp, neg = [45, 30], [50, 50], [5, 6], [50, 50]
        lo, hi = equalized_odds_gap_interval(tp, pos, fp, neg)
        eod = max(abs(45 / 50 - 30 / 50), abs(5 / 50 - 6 / 50))
        assert lo <= eod <= hi and lo > 0

    def test_contains_observed_gap_at_extreme_small_groups(self):
        lo, hi = equalized_odds_gap_interval([0, 10], [10, 10], [0, 0], [10, 10], level=0.8)
        assert lo <= 1.0 == hi

    def test_empty_label_stratum(self):
        with pytest.raises(IntervalUndefinedError) as e:
            equalized_odds_gap_interval([0, 3], [0, 10], [1, 1], [10, 10])
        assert e.value.reason == "empty_label_stratum"


class TestWelchGapInterval:
    def test_equal_means(self):
        lo, hi = welch_gap_interval([1.0, 1.0], [1.0, 1.0], [100, 100])
        assert lo == 0.0 and hi > 0

    def test_unbounded_upper(self):
        lo, hi = welch_gap_interval([0.0, 5.0], [1.0, 1.0], [50, 50])
        assert lo > 4 and hi > 5

    def test_undefined_cases(self):
        with pytest.raises(IntervalUndefinedError) as e:
            welch_gap_interval([1.0, 2.0], [0.0, 0.0], [10, 10])
        assert e.value.reason == "zero_variance"
        with pytest.raises(IntervalUndefinedError) as e:
            welch_gap_interval([1.0, 2.0], [1.0, 1.0], [1, 10])
        assert e.value.reason == "group_too_small"


class TestPermutationGapPvalue:
    def test_reproducible_and_bounded(self):
        rng = np.random.default_rng(0)
        v = rng.binomial(1, 0.5, 200).astype(float)
        g = np.repeat([0, 1], 100)
        p1 = permutation_gap_pvalue(v, g, n_permutations=500, random_state=7)
        p2 = permutation_gap_pvalue(v, g, n_permutations=500, random_state=7)
        assert p1 == p2 and 1 / 501 <= p1 <= 1

    def test_strong_effect_gives_minimum_pvalue(self):
        v = np.r_[np.ones(50), np.zeros(50)]
        g = np.repeat([0, 1], 50)
        assert permutation_gap_pvalue(v, g, n_permutations=199) == pytest.approx(1 / 200)

    def test_no_effect_gives_large_pvalue(self):
        v = np.tile([0.0, 1.0], 50)
        g = np.repeat([0, 1], 50)
        assert permutation_gap_pvalue(v, g, n_permutations=199) == 1.0

    def test_strata_shuffle_within(self):
        y_true = np.tile([1, 0], 50)
        y_pred = np.tile([1, 0], 50).astype(float)
        g = np.repeat([0, 1], 50)
        # Within each y_true stratum y_pred is constant: no shuffle can create a gap.
        assert permutation_gap_pvalue(y_pred, g, strata=y_true, n_permutations=99) == 1.0

    def test_empty_stratum_raises(self):
        with pytest.raises(IntervalUndefinedError):
            permutation_gap_pvalue([1.0, 0.0, 1.0], [0, 0, 1], strata=[1, 0, 1])

    @pytest.mark.parametrize("use_strata", [False, True])
    def test_binary_fast_path_matches_row_shuffles(self, use_strata):
        """0/1 values use count-level draws; {0, 2} values force row shuffles. The
        gap scales by 2, the null distribution does not, so p-values agree."""
        rng = np.random.default_rng(5)
        g = rng.integers(0, 3, 600)
        v = rng.binomial(1, 0.3 + 0.03 * g).astype(float)
        strata = rng.integers(0, 2, 600) if use_strata else None
        p_bin = permutation_gap_pvalue(v, g, strata=strata, n_permutations=4000)
        p_row = permutation_gap_pvalue(2 * v, g, strata=strata, n_permutations=4000)
        assert abs(p_bin - p_row) < 0.03

    def test_chunking_matches_single_pass(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=3000)
        g = rng.integers(0, 3, 3000)
        p = permutation_gap_pvalue(v, g, n_permutations=2000, random_state=3)
        assert 0 < p <= 1
