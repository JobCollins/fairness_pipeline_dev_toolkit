import numpy as np
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

def test_auto_backend_selection_runs():
    # will pick fairlearn -> aequitas -> native based on availability
    fa = FairnessAnalyzer(min_group_size=1) #auto-pick
    assert fa.backend in {"fairlearn", "aequitas", "native"}

    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    sensitive = np.array(["A", "A", "B", "B", "A", "B"])

    # both metrics should return a schema with n_per_group
    dp_res = fa.demographic_parity_difference(y_pred, sensitive)
    assert dp_res.metric == "demographic_parity_difference"
    assert isinstance(dp_res.n_per_group, dict)

    eo_res = fa.equalized_odds_difference(y_true, y_pred, sensitive)
    assert eo_res.metric == "equalized_odds_difference"
    assert isinstance(eo_res.n_per_group, dict)

def test_force_native_backend():
    fa = FairnessAnalyzer(min_group_size=1, backend="native")
    assert fa.backend == "native"

    # y_true = np.array([0, 1, 0, 1, 0, 1])
    # y_pred = np.array([0, 1, 0, 0, 1, 1])
    # sensitive = np.array(["A", "A", "B", "B", "A", "B"])

    # dp_res = fa.demographic_parity_difference(y_pred, sensitive)
    # assert dp_res.metric == "demographic_parity_difference"
    # assert isinstance(dp_res.n_per_group, dict)

    # eo_res = fa.equalized_odds_difference(y_true, y_pred, sensitive)
    # assert eo_res.metric == "equalized_odds_difference"
    # assert isinstance(eo_res.n_per_group, dict)