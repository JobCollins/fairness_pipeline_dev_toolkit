import numpy as np
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

def test_result_schema_has_required_fields():
    fa = FairnessAnalyzer(min_group_size=1, backend="native") #auto-pick
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    sensitive = np.array(["A", "A", "B", "B", "A", "B"])

    dp_res = fa.demographic_parity_difference(y_pred, sensitive)
    assert hasattr(dp_res, "metric")
    assert hasattr(dp_res, "value")
    assert hasattr(dp_res, "ci")
    assert hasattr(dp_res, "effect_size")
    assert hasattr(dp_res, "n_per_group")

    eo_res = fa.equalized_odds_difference(y_true, y_pred, sensitive)
    assert hasattr(eo_res, "metric")
    assert hasattr(eo_res, "value")
    assert hasattr(eo_res, "ci")
    assert hasattr(eo_res, "effect_size")
    assert hasattr(eo_res, "n_per_group")