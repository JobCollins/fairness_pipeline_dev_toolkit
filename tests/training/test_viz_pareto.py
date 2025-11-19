import matplotlib
import numpy as np

# Use non-interactive backend for tests to prevent plot windows from opening
matplotlib.use("Agg")

from fairness_pipeline_dev_toolkit.training import plot_pareto, sweep_pareto


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


def test_sweep_pareto_with_save_path(tmp_path):
    """Test that sweep_pareto saves plot when save_path is provided."""
    rng = np.random.RandomState(0)
    n, d = 150, 6
    X = rng.randn(n, d)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.3 * s + rng.randn(n) * 0.2) > 0).astype(int)

    # split
    Xtr, Xv = X[:100], X[100:]
    ytr, yv = y[:100], y[100:]
    str_, sv = s[:100], s[100:]

    save_path = tmp_path / "pareto_test.png"

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
        save_path=str(save_path),
    )

    # Verify points are returned
    assert isinstance(pts, list) and len(pts) == 2
    assert {"eta", "accuracy", "dp_diff"}.issubset(pts[0].keys())

    # Verify plot file was created
    assert save_path.exists(), f"Plot file should be created at {save_path}"
    assert save_path.stat().st_size > 0, "Plot file should not be empty"


def test_sweep_pareto_without_save_path():
    """Test that sweep_pareto works without save_path (no file created)."""
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
        save_path=None,  # Explicitly None
    )

    assert isinstance(pts, list) and len(pts) == 2
    assert {"eta", "accuracy", "dp_diff"}.issubset(pts[0].keys())


def test_plot_pareto_with_save_path(tmp_path):
    """Test plot_pareto saves file when save_path is provided."""
    # Create mock pareto points
    points = [
        {"eta": 0.0, "accuracy": 0.85, "dp_diff": 0.15},
        {"eta": 0.5, "accuracy": 0.82, "dp_diff": 0.08},
        {"eta": 1.0, "accuracy": 0.78, "dp_diff": 0.05},
    ]

    save_path = tmp_path / "plot_test.png"
    plot_pareto(points, save_path=str(save_path))

    # Verify file was created
    assert save_path.exists(), f"Plot file should be created at {save_path}"
    assert save_path.stat().st_size > 0, "Plot file should not be empty"


def test_plot_pareto_without_save_path():
    """Test plot_pareto works without save_path (no file created, just displays)."""
    points = [
        {"eta": 0.0, "accuracy": 0.85, "dp_diff": 0.15},
        {"eta": 0.5, "accuracy": 0.82, "dp_diff": 0.08},
    ]

    # Should not raise an error
    plot_pareto(points, save_path=None)


def test_sweep_pareto_multiple_etas():
    """Test sweep_pareto with multiple eta values."""
    rng = np.random.RandomState(0)
    n, d = 150, 6
    X = rng.randn(n, d)
    s = (rng.rand(n) > 0.5).astype(int)
    y = ((X[:, 0] + 0.3 * s + rng.randn(n) * 0.2) > 0).astype(int)

    Xtr, Xv = X[:100], X[100:]
    ytr, yv = y[:100], y[100:]
    str_, sv = s[:100], s[100:]

    etas = [0.0, 0.1, 0.5, 1.0, 2.0]
    pts = sweep_pareto(Xtr, ytr, str_, Xv, yv, sv, etas=etas, epochs=3, lr=1e-3, device="cpu")

    assert len(pts) == len(etas)
    for pt in pts:
        assert {"eta", "accuracy", "dp_diff"}.issubset(pt.keys())
        assert pt["eta"] in etas
        assert 0.0 <= pt["accuracy"] <= 1.0
        assert pt["dp_diff"] >= 0.0
