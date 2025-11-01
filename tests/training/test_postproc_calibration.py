import numpy as np

from fairness_pipeline_dev_toolkit.training import GroupFairnessCalibrator


def _toy_scores_labels_groups(n=120, seed=0):
    rng = np.random.RandomState(seed)
    scores = rng.rand(n)
    groups = rng.choice([0, 1], size=n)
    # make labels somewhat consistent with scores but skew by group
    logits = scores + 0.2 * (groups == 1) + 0.05 * rng.randn(n)
    probs = 1.0 / (1.0 + np.exp(-5 * (logits - 0.5)))
    labels = (rng.rand(n) < probs).astype(int)
    return scores, labels, groups


def test_group_calibrator_platt():
    scores, labels, groups = _toy_scores_labels_groups()
    cal = GroupFairnessCalibrator(method="platt", min_samples=20).fit(scores, labels, groups)
    out = cal.transform(scores, groups)
    assert out.shape == scores.shape
    assert np.all((out >= 0) & (out <= 1))


def test_group_calibrator_isotonic():
    scores, labels, groups = _toy_scores_labels_groups(seed=1)
    cal = GroupFairnessCalibrator(method="isotonic", min_samples=20).fit(scores, labels, groups)
    out = cal.transform(scores, groups)
    assert out.shape == scores.shape
    assert np.all((out >= 0) & (out <= 1))
