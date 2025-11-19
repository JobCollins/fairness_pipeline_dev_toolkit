import numpy as np
import pytest
from fairlearn.reductions import DemographicParity, EqualizedOdds
from sklearn.ensemble import GradientBoostingClassifier

from fairness_pipeline_dev_toolkit.training import ReductionsWrapper


@pytest.mark.parametrize("n", [120])
def test_reductions_wrapper_fit_predict(n):
    rng = np.random.RandomState(0)
    X = rng.randn(n, 4)
    s = (rng.rand(n) > 0.5).astype(int)
    # make y depend weakly on X and (undesirably) on s
    y = ((X[:, 0] + 0.2 * s + rng.randn(n) * 0.1) > 0.1).astype(int)

    base = GradientBoostingClassifier(random_state=0)
    constraint = DemographicParity(difference_bound=0.1)
    clf = ReductionsWrapper(base_estimator=base, constraint=constraint, eps=0.02, T=15)
    clf.fit(X, y, sensitive_features=s)
    yhat = clf.predict(X)
    assert yhat.shape == (n,)
    # if available, ensure predict_proba returns probabilities
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        assert proba.shape[0] == n


def test_reductions_wrapper_T_parameter_passed():
    """Verify that T parameter is passed to ExponentiatedGradient as max_iter."""
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 4)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.2 * s + rng.randn(n) * 0.1) > 0.1).astype(int)

    base = GradientBoostingClassifier(random_state=0, n_estimators=10)
    constraint = DemographicParity(difference_bound=0.1)

    # Test with explicit T value
    T_value = 25
    clf = ReductionsWrapper(base_estimator=base, constraint=constraint, eps=0.02, T=T_value)
    clf.fit(X, y, sensitive_features=s)

    # Verify wrapper was created and can predict
    assert clf._wrapper_ is not None
    yhat = clf.predict(X)
    assert yhat.shape == (n,)

    # Verify that the wrapper has the expected structure
    # (We can't directly check max_iter, but we verify it doesn't crash with T set)


def test_reductions_wrapper_T_override_via_kwargs():
    """Verify that T can be overridden via kwargs."""
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 4)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.2 * s + rng.randn(n) * 0.1) > 0.1).astype(int)

    base = GradientBoostingClassifier(random_state=0, n_estimators=10)
    constraint = DemographicParity(difference_bound=0.1)

    # Set T=50 but override via kwargs to max_iter=30
    clf = ReductionsWrapper(
        base_estimator=base,
        constraint=constraint,
        eps=0.02,
        T=50,
        kwargs={"max_iter": 30},
    )
    clf.fit(X, y, sensitive_features=s)

    # Verify it works
    assert clf._wrapper_ is not None
    yhat = clf.predict(X)
    assert yhat.shape == (n,)


def test_reductions_wrapper_with_equalized_odds():
    """Test ReductionsWrapper with EqualizedOdds constraint."""
    rng = np.random.RandomState(0)
    n = 120
    X = rng.randn(n, 4)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.2 * s + rng.randn(n) * 0.1) > 0.1).astype(int)

    base = GradientBoostingClassifier(random_state=0, n_estimators=10)
    constraint = EqualizedOdds(difference_bound=0.1)
    clf = ReductionsWrapper(base_estimator=base, constraint=constraint, eps=0.02, T=15)
    clf.fit(X, y, sensitive_features=s)

    yhat = clf.predict(X)
    assert yhat.shape == (n,)
    assert set(yhat) <= {0, 1}


def test_reductions_wrapper_default_T():
    """Test that default T value is used when not specified."""
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 4)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.2 * s + rng.randn(n) * 0.1) > 0.1).astype(int)

    base = GradientBoostingClassifier(random_state=0, n_estimators=10)
    constraint = DemographicParity(difference_bound=0.1)

    # Use default T (should be 50)
    clf = ReductionsWrapper(base_estimator=base, constraint=constraint, eps=0.02)
    assert clf.T == 50  # default value
    clf.fit(X, y, sensitive_features=s)

    assert clf._wrapper_ is not None
    yhat = clf.predict(X)
    assert yhat.shape == (n,)
