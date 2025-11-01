import numpy as np

from fairness_pipeline_dev_toolkit.training import sweep_pareto


def test_sweep_pareto_runs():
    rng = np.random.RandomState(0)
    n, d = 150, 6
    X = rng.randn(n, d)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.3 * s + rng.randn(n) * 0.2) > 0).astype(int)

    # split
    Xtr, Xv = X[:100], X[100:]
    ytr, yv = y[:100], y[100:]
    str_, sv = s[:100], s[100:]

    pts = sweep_pareto(
        Xtr,
        ytr,
        str_,
        Xv,
        yv,
        sv,
        etas=(0.0, 0.2),
        epochs=3,
        lr=1e-3,
        device="cpu",
    )
    assert isinstance(pts, list) and len(pts) == 2
    assert {"eta", "accuracy", "dp_diff"}.issubset(pts[0].keys())
