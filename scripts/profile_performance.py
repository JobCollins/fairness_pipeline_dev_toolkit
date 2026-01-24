#!/usr/bin/env python3
"""
Performance profiling script for critical paths.

This script uses cProfile to identify bottlenecks in critical operations.
Run with: python scripts/profile_performance.py
"""

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
    run_detectors,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci


def make_test_data(n: int = 50_000, seed: int = 42):
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


def profile_metrics_computation():
    """Profile fairness metrics computation."""
    print("\n" + "=" * 60)
    print("Profiling: Metrics Computation")
    print("=" * 60)

    y_true, y_pred, sensitive, _ = make_test_data(n=50_000)
    analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile demographic parity
    result_dp = analyzer.demographic_parity_difference(y_pred, sensitive, with_ci=False)

    # Profile equalized odds
    result_eo = analyzer.equalized_odds_difference(y_true, y_pred, sensitive, with_ci=False)

    profiler.disable()

    # Print top 20 functions by cumulative time
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(s.getvalue())

    print(f"Demographic Parity Result: {result_dp.value:.4f}")
    print(f"Equalized Odds Result: {result_eo.value:.4f}")


def profile_bootstrap_ci():
    """Profile bootstrap confidence interval computation."""
    print("\n" + "=" * 60)
    print("Profiling: Bootstrap CI Computation")
    print("=" * 60)

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 10_000)

    def mean_stat(x):
        return np.mean(x)

    profiler = cProfile.Profile()
    profiler.enable()

    ci = bootstrap_ci(data=data, stat_fn=mean_stat, B=1000, method="percentile", random_state=42)

    profiler.disable()

    # Print top 20 functions by cumulative time
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(s.getvalue())

    print(f"Bootstrap CI Result: [{ci[0]:.4f}, {ci[1]:.4f}]")


def profile_pipeline_operations():
    """Profile pipeline operations."""
    print("\n" + "=" * 60)
    print("Profiling: Pipeline Operations")
    print("=" * 60)

    _, _, _, df = make_test_data(n=10_000)
    config_dict = {
        "sensitive": ["sensitive"],
        "features": ["feature_1", "feature_2"],
        "target": "y",
        "pipeline": [
            {
                "name": "reweigh",
                "transformer": "InstanceReweighting",
            }
        ],
    }
    config = PipelineConfig(**config_dict)

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile detector execution
    detector_report = run_detectors(df=df, cfg=config)

    # Profile pipeline build and apply
    pipeline = build_pipeline(config)
    transformed_df, metadata = apply_pipeline(pipeline, df)

    profiler.disable()

    # Print top 20 functions by cumulative time
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(s.getvalue())

    print(f"Detector report: {len(detector_report.body)} detections")
    print(f"Transformed shape: {transformed_df.shape}")


def profile_intersectional_analysis():
    """Profile intersectional analysis."""
    print("\n" + "=" * 60)
    print("Profiling: Intersectional Analysis")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 50_000
    y_pred = rng.integers(0, 2, size=n)

    attrs = pd.DataFrame(
        {
            "race": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
            "gender": rng.choice(["M", "F"], size=n, p=[0.6, 0.4]),
        }
    )

    analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

    profiler = cProfile.Profile()
    profiler.enable()

    result = analyzer.demographic_parity_difference(
        y_pred, sensitive=None, intersectional=True, attrs_df=attrs, with_ci=False
    )

    profiler.disable()

    # Print top 20 functions by cumulative time
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(s.getvalue())

    print(f"Intersectional Result: {result.value:.4f}")


def main():
    """Run all profiling operations."""
    print("Performance Profiling Report")
    print("=" * 60)
    print("This script profiles critical paths to identify bottlenecks.")
    print("=" * 60)

    profile_metrics_computation()
    profile_bootstrap_ci()
    profile_pipeline_operations()
    profile_intersectional_analysis()

    print("\n" + "=" * 60)
    print("Profiling Complete")
    print("=" * 60)
    print("\nTo save profile data to a file, use:")
    print("  python -m cProfile -o profile.stats scripts/profile_performance.py")
    print("Then analyze with:")
    print("  python -m pstats profile.stats")


if __name__ == "__main__":
    main()
