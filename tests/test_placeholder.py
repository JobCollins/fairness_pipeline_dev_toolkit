from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer


def test_placeholder():
    assert True, "CI pipeline operational."

def test_placeholder_runs():
    fa = FairnessAnalyzer(min_group_size=1)
    y_pred = [0, 1, 1, 0]
    sensitive = ["A", "A", "B", "B"]
    res = fa.demographic_parity_difference(y_pred, sensitive)
    assert res.metric == "demographic_parity_difference"