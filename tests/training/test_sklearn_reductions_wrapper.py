import numpy as np
import pytest
from fairlearn.reductions import DemographicParity
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
