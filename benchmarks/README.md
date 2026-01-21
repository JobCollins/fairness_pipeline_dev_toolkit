# Performance Benchmarks

This directory contains performance benchmarks for critical components of the Fairness Pipeline Development Toolkit.

## Available Benchmarks

### `benchmark_metrics_100k.py`
Benchmarks fairness metrics computation (demographic parity, equalized odds, MAE parity) with and without intersectional analysis.

**Usage:**
```bash
python benchmarks/benchmark_metrics_100k.py
```

**What it measures:**
- Time to compute demographic parity difference
- Time to compute equalized odds difference
- Time to compute MAE parity difference
- Performance with intersectional analysis

### `benchmark_pipeline.py`
Benchmarks pipeline operations including bias detection and transformation.

**Usage:**
```bash
python benchmarks/benchmark_pipeline.py
```

**What it measures:**
- Time to run bias detectors
- Time to build pipeline
- Time to apply pipeline transformations
- Performance across different dataset sizes (10k, 50k, 100k)

### `benchmark_bootstrap.py`
Benchmarks bootstrap confidence interval computation, which can be computationally expensive.

**Usage:**
```bash
python benchmarks/benchmark_bootstrap.py
```

**What it measures:**
- Time to compute bootstrap CIs
- Performance with different bootstrap sample counts
- Performance with different data sizes
- Comparison of percentile vs BCa methods

## Running All Benchmarks

To run all benchmarks:

```bash
# From project root
python benchmarks/benchmark_metrics_100k.py
python benchmarks/benchmark_pipeline.py
python benchmarks/benchmark_bootstrap.py
```

## Interpreting Results

- **Throughput**: Samples processed per second
- **Scalability**: How performance changes with dataset size
- **Bottlenecks**: Identify slow operations that need optimization

## Integration with CI/CD

These benchmarks can be integrated into CI/CD pipelines to:
- Track performance regressions
- Ensure performance doesn't degrade with new changes
- Set performance baselines for releases

## Future Enhancements

- Automated performance regression detection
- Performance comparison across versions
- Memory usage profiling
- Parallel processing benchmarks
