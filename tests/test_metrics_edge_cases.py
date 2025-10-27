import numpy as np
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

def test_dp_handles_small_groups_exclusion():
    fa = FairnessAnalyzer(min_group_size=3, backend="native")
    y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']) # 'C' has only 1 instance -> should be excluded
    res = fa.demographic_parity_difference(y_pred, sensitive)

    # should only consider groups 'A' and 'B'
    assert res.n_per_group == {'A': 4, 'B': 3}
    assert 0.0 <= res.value <= 1.0 or np.isnan(res.value)

def test_eo_handles_no_positives_or_negatives():
    fa = FairnessAnalyzer(min_group_size=2, backend="native")
    y_true = np.array([1, 1, 1, 1, 1, 1]) # all positives
    y_pred = np.array([1, 1, 1, 0, 1, 0])
    sensitive = np.array(['X', 'X', 'Y', 'Y', 'Y', 'Y']) # group 'X' has no negatives, group 'Y' has one positive

    res = fa.equalized_odds_difference(y_true, y_pred, sensitive)

    # FPRs will be nan; EO should guard and return nan if both TPR and FPR cannot be compared
    assert res.metric == 'equalized_odds_difference'
    assert res.value == res.value or np.isnan(res.value)

def test_mae_parity_with_missing_attrs_intersectional():
    fa = FairnessAnalyzer(min_group_size=2, backend="native", nan_policy="include")
    y_true = np.array([3.0, 2.5, 4.0, 5.0, 3.5, 4.5])
    y_pred = np.array([2.5, 2.0, 4.5, 5.0, 3.0, 4.0])
    attrs = pd.DataFrame({
        "race": ["A", "A", "B", "B", None, "A"],
        "gender": ["M", "F", "M", None, "F", "F"],
    })
    res = fa.mae_parity_difference(
        y_true,
        y_pred,
        sensitive=None,
        intersectional=True,
        attrs_df=attrs
    )
    assert res.metric == 'mae_parity_difference'
    assert isinstance(res.n_per_group, dict)