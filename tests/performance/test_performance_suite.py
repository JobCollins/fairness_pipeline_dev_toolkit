"""
Performance test suite for critical paths.

This test suite establishes performance baselines and detects regressions.
Tests are designed to run in CI/CD to track performance over time.
"""

import time

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
    run_detectors,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci


def make_test_data(n: int = 10_000, seed: int = 42) -> tuple:
    """Generate synthetic test data."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    sensitive = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

    df = pd.DataFrame(
        {
            "feature_1": rng.normal(0, 1, n),
            "feature_2": rng.normal(0, 1, n),
            "sensitive": sensitive,
            "y": y_true,
        }
    )

    return y_true, y_pred, sensitive, df


class TestMetricsPerformance:
    """Performance tests for fairness metrics computation."""

    def test_demographic_parity_performance(self):
        """Test demographic parity computation performance."""
        _, y_pred, sensitive, _ = make_test_data(n=50_000)
        analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

        start = time.time()
        result = analyzer.demographic_parity_difference(y_pred, sensitive, with_ci=False)
        elapsed = time.time() - start

        # Baseline: should complete in < 2 seconds for 50k samples
        assert elapsed < 2.0, f"Demographic parity took {elapsed:.2f}s, expected < 2.0s"
        assert result.value is not None
        _ = result  # Used in assertion above

    def test_equalized_odds_performance(self):
        """Test equalized odds computation performance."""
        y_true, y_pred, sensitive, _ = make_test_data(n=50_000)
        analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

        start = time.time()
        result = analyzer.equalized_odds_difference(y_true, y_pred, sensitive, with_ci=False)
        elapsed = time.time() - start

        # Baseline: should complete in < 3 seconds for 50k samples
        assert elapsed < 3.0, f"Equalized odds took {elapsed:.2f}s, expected < 3.0s"
        assert result.value is not None

    def test_mae_parity_performance(self):
        """Test MAE parity computation performance."""
        rng = np.random.default_rng(42)
        y_reg = rng.normal(0, 1, 50_000)
        y_hat = y_reg + rng.normal(0, 0.5, 50_000)
        sensitive = rng.choice(["A", "B", "C"], size=50_000, p=[0.5, 0.3, 0.2])
        analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

        start = time.time()
        result = analyzer.mae_parity_difference(y_reg, y_hat, sensitive, with_ci=False)
        elapsed = time.time() - start

        # Baseline: should complete in < 2 seconds for 50k samples
        assert elapsed < 2.0, f"MAE parity took {elapsed:.2f}s, expected < 2.0s"
        assert result.value is not None


class TestBootstrapPerformance:
    """Performance tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_performance(self):
        """Test bootstrap CI computation performance."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10_000)

        def mean_stat(x):
            return np.mean(x)

        start = time.time()
        ci = bootstrap_ci(
            data=data,
            stat_fn=mean_stat,
            B=500,  # Reduced for faster tests
            method="percentile",
            random_state=42,
        )
        elapsed = time.time() - start

        # Baseline: should complete in < 10 seconds for 10k samples with 500 bootstrap
        assert elapsed < 10.0, f"Bootstrap CI took {elapsed:.2f}s, expected < 10.0s"
        assert len(ci) == 2
        assert ci[0] < ci[1]


class TestPipelinePerformance:
    """Performance tests for pipeline operations."""

    def test_detector_performance(self):
        """Test bias detector performance."""
        _, _, _, df = make_test_data(n=10_000)
        config = PipelineConfig(sensitive=["sensitive"])

        start = time.time()
        report = run_detectors(df=df, cfg=config)
        elapsed = time.time() - start

        # Baseline: should complete in < 3 seconds for 10k samples
        assert elapsed < 3.0, f"Detector execution took {elapsed:.2f}s, expected < 3.0s"
        assert report is not None

    def test_pipeline_build_and_apply_performance(self):
        """Test pipeline build and apply performance."""
        _, _, _, df = make_test_data(n=10_000)
        from fairness_pipeline_dev_toolkit.pipeline.config import PipelineStep

        config = PipelineConfig(
            sensitive=["sensitive"],
            pipeline=[
                PipelineStep(
                    name="reweigh",
                    transformer="InstanceReweighting",
                )
            ],
        )

        # Build pipeline
        start = time.time()
        pipeline = build_pipeline(config)
        build_time = time.time() - start

        # Apply pipeline
        start = time.time()
        transformed_df, metadata = apply_pipeline(pipeline, df)
        apply_time = time.time() - start

        total_time = build_time + apply_time

        # Baseline: should complete in < 2 seconds for 10k samples
        assert total_time < 2.0, f"Pipeline build+apply took {total_time:.2f}s, expected < 2.0s"
        assert transformed_df is not None
        assert transformed_df.shape[0] == df.shape[0]


class TestScalability:
    """Tests for scalability across different data sizes."""

    @pytest.mark.parametrize("n", [1_000, 10_000, 50_000])
    def test_metrics_scalability(self, n):
        """Test that metrics computation scales linearly."""
        _, y_pred, sensitive, _ = make_test_data(n=n)
        analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

        start = time.time()
        result = analyzer.demographic_parity_difference(y_pred, sensitive, with_ci=False)
        elapsed = time.time() - start

        # Should scale roughly linearly: 50k should be ~5x slower than 10k
        # But allow some overhead, so use a more lenient check
        max_time = (n / 1_000) * 0.1  # 0.1s per 1k samples
        assert elapsed < max_time, f"Scalability issue: {n} samples took {elapsed:.2f}s"
        assert result.value is not None  # Verify result is valid


def pytest_configure(config):
    """Configure pytest for performance tests."""
    # Mark all tests in this module as performance tests
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
