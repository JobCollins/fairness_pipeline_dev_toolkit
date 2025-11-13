import numpy as np

from fairness_pipeline_dev_toolkit.metrics.core import FairnessAnalyzer


def _binary_groups():
    y_pred = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int)
    sensitive = np.array(["A"] * 10 + ["B"] * 10)
    return y_pred, sensitive


def test_demographic_parity_defaults_include_statistics():
    analyzer = FairnessAnalyzer(min_group_size=5, backend="native")
    y_pred, sensitive = _binary_groups()

    np.random.seed(0)
    result = analyzer.demographic_parity_difference(
        y_pred=y_pred, sensitive=sensitive, ci_samples=200
    )

    assert result.ci is not None
    assert np.all(np.isfinite(result.ci))
    assert result.effect_size is not None
    assert np.isfinite(result.effect_size)


def test_equalized_odds_defaults_include_statistics():
    analyzer = FairnessAnalyzer(min_group_size=5, backend="native")
    y_pred, sensitive = _binary_groups()
    y_true = np.array(
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
        dtype=int,
    )

    np.random.seed(0)
    result = analyzer.equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive=sensitive, ci_samples=200
    )

    assert result.ci is not None
    assert np.all(np.isfinite(result.ci))
    assert result.effect_size is not None
    assert np.isfinite(result.effect_size)


def test_mae_parity_defaults_include_statistics():
    analyzer = FairnessAnalyzer(min_group_size=5, backend="native")
    sensitive = np.array(["A"] * 10 + ["B"] * 10)
    y_true = np.concatenate([np.linspace(0.1, 0.9, 10), np.linspace(0.1, 0.9, 10)])
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=0.05, size=20)
    y_pred = y_true + np.concatenate([0.1 * np.ones(10), -0.05 * np.ones(10)]) + noise

    np.random.seed(0)
    result = analyzer.mae_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive=sensitive, ci_samples=200
    )

    assert result.ci is not None
    assert np.all(np.isfinite(result.ci))
    assert result.effect_size is not None
    assert np.isfinite(result.effect_size)
