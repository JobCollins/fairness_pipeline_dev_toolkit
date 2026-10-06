"""Unit tests for C2b template Bonferroni-t intervals (Wave 3a / BL-016)."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.stats.template_intervals import (
    T_MIN_TEMPLATES,
    T_SMALL_NOTE_UNTIL,
    small_template_note,
    template_bonferroni_t_max_mean,
)


def test_t_min_constant_and_notes():
    assert T_MIN_TEMPLATES == 5
    assert T_SMALL_NOTE_UNTIL == 10
    assert small_template_note(5).startswith("small_T:")
    assert small_template_note(10).startswith("small_T:")
    assert small_template_note(11) is None


def test_too_few_templates_undefined():
    with pytest.raises(IntervalUndefinedError) as e:
        template_bonferroni_t_max_mean({"a": [0.1, 0.2, 0.15], "b": [0.0, 0.1, 0.05]})
    assert e.value.reason == "too_few_templates"


def test_contains_max_mean_and_zero_variance_collapses():
    rng = np.random.default_rng(0)
    arms = {"gender": rng.normal(0.2, 0.05, 12), "age": rng.normal(0.1, 0.05, 12)}
    lo, hi = template_bonferroni_t_max_mean(arms)
    point = max(float(v.mean()) for v in arms.values())
    assert lo <= point <= hi
    # Identical templates → exact mean known → degenerate interval at the point.
    flat = {"a": np.full(8, 0.3), "b": np.full(8, 0.1)}
    assert template_bonferroni_t_max_mean(flat) == (0.3, 0.3)


def test_wider_at_higher_level():
    rng = np.random.default_rng(1)
    arms = {"d": rng.normal(size=20)}
    lo90, hi90 = template_bonferroni_t_max_mean(arms, level=0.90)
    lo99, hi99 = template_bonferroni_t_max_mean(arms, level=0.99)
    assert lo99 <= lo90 and hi99 >= hi90
