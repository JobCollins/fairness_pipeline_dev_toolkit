"""
Performance benchmarks for pipeline operations.

This module benchmarks the performance of bias detection and mitigation
transformers in the pipeline module.
"""

import time

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline import apply_pipeline, build_pipeline
from fairness_pipeline_dev_toolkit.pipeline.detectors import run_detectors


def make_fake_data(n=50_000, seed=42):
    """Generate synthetic data for benchmarking."""
    rng = np.random.default_rng(seed)

    # Create DataFrame with features and sensitive attributes
    df = pd.DataFrame(
        {
            "feature_1": rng.normal(0, 1, n),
            "feature_2": rng.normal(0, 1, n),
            "feature_3": rng.integers(0, 10, n),
            "sensitive": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
            "y": rng.integers(0, 2, n),
        }
    )

    return df


def benchmark_detectors(df, config):
    """Benchmark bias detection performance."""
    print(f"\n=== Benchmarking Detectors (n={len(df)}) ===")

    start = time.time()
    detector_report = run_detectors(df=df, cfg=config)
    end = time.time()

    print(f"Detector execution time: {end - start:.3f} seconds")
    print(f"Detections found: {len(detector_report.body)}")

    return end - start


def benchmark_pipeline(df, config):
    """Benchmark pipeline transformation performance."""
    print(f"\n=== Benchmarking Pipeline Transformation (n={len(df)}) ===")

    # Build pipeline
    start = time.time()
    pipeline = build_pipeline(config)
    build_time = time.time() - start
    print(f"Pipeline build time: {build_time:.3f} seconds")

    # Apply pipeline
    start = time.time()
    transformed_df, metadata = apply_pipeline(pipeline, df)
    transform_time = time.time() - start

    print(f"Pipeline transformation time: {transform_time:.3f} seconds")
    print(f"Output shape: {transformed_df.shape}")

    return build_time + transform_time


if __name__ == "__main__":
    # Create sample data
    df_small = make_fake_data(n=10_000)
    df_medium = make_fake_data(n=50_000)
    df_large = make_fake_data(n=100_000)

    # Create minimal config
    config_dict = {
        "sensitive": ["sensitive"],
        "pipeline": [
            {"name": "reweigh", "transformer": "InstanceReweighting"},
            {
                "name": "repair",
                "transformer": "DisparateImpactRemover",
                "params": {"features": ["feature_1", "feature_2"], "repair_level": 0.8},
            },
        ],
    }

    # Convert to config object (simplified for benchmark)
    from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig

    config = PipelineConfig(**config_dict)

    # Run benchmarks
    for df, size in [(df_small, "Small"), (df_medium, "Medium"), (df_large, "Large")]:
        print(f"\n{'='*60}")
        print(f"{size} Dataset Benchmark")
        print(f"{'='*60}")

        benchmark_detectors(df.copy(), config)
        benchmark_pipeline(df.copy(), config)
