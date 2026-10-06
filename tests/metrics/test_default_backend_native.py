"""Default backend is always native (BL-027)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.metrics.fairlearn_adapter import FairlearnAdapter


def test_default_backend_is_native_even_if_fairlearn_available():
    """Installing / mocking Fairlearn must not change FairnessAnalyzer().backend."""
    with patch.object(FairlearnAdapter, "available", return_value=True):
        fa = FairnessAnalyzer(min_group_size=1)
        assert fa.backend == "native"


def test_backend_none_means_native():
    fa = FairnessAnalyzer(min_group_size=1, backend=None)
    assert fa.backend == "native"


def test_explicit_fairlearn_backend_when_available():
    if not FairlearnAdapter().available():
        return
    fa = FairnessAnalyzer(min_group_size=1, backend="fairlearn")
    assert fa.backend == "fairlearn"
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    sensitive = np.array(["A", "A", "B", "B", "A", "B"])
    res = fa.demographic_parity_difference(y_pred, sensitive, with_ci=False)
    assert res.metric == "demographic_parity_difference"
