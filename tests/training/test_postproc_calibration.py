import numpy as np
import pytest

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


def test_group_calibrator_small_groups_skipped():
    """Test that groups smaller than min_samples are skipped."""
    rng = np.random.RandomState(0)
    n = 50
    scores = rng.rand(n)
    # Create groups where one is too small
    groups = np.array([0] * 45 + [1] * 5)  # Group 1 has only 5 samples
    labels = rng.randint(0, 2, n)

    cal = GroupFairnessCalibrator(method="platt", min_samples=20).fit(scores, labels, groups)
    # Group 1 should not be calibrated (too small)
    assert 1 not in cal.calibrators
    # Group 0 should be calibrated
    assert 0 in cal.calibrators

    out = cal.transform(scores, groups)
    assert out.shape == scores.shape
    # Group 1 should have original scores (not calibrated)
    assert np.allclose(out[groups == 1], scores[groups == 1])


def test_group_calibrator_missing_group_in_transform():
    """Test transform with groups not seen during fit."""
    scores, labels, groups = _toy_scores_labels_groups()
    cal = GroupFairnessCalibrator(method="platt", min_samples=20).fit(scores, labels, groups)

    # Transform with a new group not seen during fit
    new_scores = np.array([0.5, 0.6, 0.7])
    new_groups = np.array([0, 1, 2])  # Group 2 not seen during fit

    out = cal.transform(new_scores, new_groups)
    assert out.shape == new_scores.shape
    # Group 2 should have original scores (not calibrated)
    assert np.allclose(out[new_groups == 2], new_scores[new_groups == 2])


def test_group_calibrator_invalid_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="method must be one of"):
        GroupFairnessCalibrator(method="invalid_method")


def test_group_calibrator_empty_groups():
    """Test with empty groups array."""
    scores = np.array([0.5, 0.6, 0.7])
    labels = np.array([0, 1, 0])
    groups = np.array([])

    cal = GroupFairnessCalibrator(method="platt", min_samples=1)
    # Should handle empty groups gracefully
    cal.fit(scores[:0], labels[:0], groups)
    assert len(cal.calibrators) == 0


def test_group_calibrator_single_group():
    """Test with only one group."""
    rng = np.random.RandomState(0)
    n = 50
    scores = rng.rand(n)
    groups = np.zeros(n)  # All same group
    labels = rng.randint(0, 2, n)

    cal = GroupFairnessCalibrator(method="platt", min_samples=20).fit(scores, labels, groups)
    assert len(cal.calibrators) == 1
    assert 0 in cal.calibrators

    out = cal.transform(scores, groups)
    assert out.shape == scores.shape
    assert np.all((out >= 0) & (out <= 1))


def test_group_calibrator_multiple_groups():
    """Test with more than 2 groups."""
    rng = np.random.RandomState(0)
    n = 150
    scores = rng.rand(n)
    groups = rng.choice([0, 1, 2, 3], size=n)
    labels = rng.randint(0, 2, n)

    cal = GroupFairnessCalibrator(method="isotonic", min_samples=20).fit(scores, labels, groups)
    out = cal.transform(scores, groups)
    assert out.shape == scores.shape
    assert np.all((out >= 0) & (out <= 1))
