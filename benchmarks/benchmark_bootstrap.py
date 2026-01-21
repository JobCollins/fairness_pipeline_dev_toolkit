"""
Performance benchmarks for bootstrap confidence interval computation.

This module benchmarks the performance of bootstrap CI computation,
which can be computationally expensive for large datasets.
"""

import time

import numpy as np

from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci


def make_fake_data(n=100_000, seed=42):
    """Generate synthetic data for benchmarking."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)


def benchmark_bootstrap_ci(data, confidence_level=0.95, n_bootstrap=1000, method="percentile"):
    """Benchmark bootstrap CI computation."""
    print("\n=== Benchmarking Bootstrap CI ===")
    print(f"Data size: {len(data):,}")
    print(f"Bootstrap samples: {n_bootstrap:,}")
    print(f"Method: {method}")
    print(f"Confidence level: {confidence_level}")

    start = time.time()
    ci = bootstrap_ci(
        data=data,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        method=method,
        random_state=42,
    )
    end = time.time()

    elapsed = end - start
    print(f"Computation time: {elapsed:.3f} seconds")
    print(f"CI result: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Throughput: {len(data) / elapsed:,.0f} samples/second")

    return elapsed


if __name__ == "__main__":
    # Test different data sizes
    sizes = [10_000, 50_000, 100_000, 500_000]

    # Test different bootstrap sample counts
    bootstrap_counts = [100, 500, 1000, 2000]

    print("=" * 60)
    print("Bootstrap CI Performance Benchmarks")
    print("=" * 60)

    for size in sizes:
        data = make_fake_data(n=size)
        print(f"\n{'='*60}")
        print(f"Data Size: {size:,}")
        print(f"{'='*60}")

        for n_bootstrap in bootstrap_counts:
            try:
                benchmark_bootstrap_ci(data=data, n_bootstrap=n_bootstrap, method="percentile")
            except Exception as e:
                print(f"Error with n_bootstrap={n_bootstrap}: {e}")

        # Test BCa method (more computationally expensive)
        print("\n--- Testing BCa method (more expensive) ---")
        try:
            benchmark_bootstrap_ci(data=data, n_bootstrap=1000, method="bca")
        except Exception as e:
            print(f"Error with BCa method: {e}")
