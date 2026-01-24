# Performance Guide

This document provides performance benchmarks, optimization tips, and best practices for using the Fairness Pipeline Development Toolkit efficiently.

---

## Table of Contents

- [Performance Benchmarks](#performance-benchmarks)
- [Performance Characteristics](#performance-characteristics)
- [Optimization Tips](#optimization-tips)
- [Scalability Considerations](#scalability-considerations)
- [Memory Usage](#memory-usage)
- [CI/CD Integration](#cicd-integration)

---

## Performance Benchmarks

### Benchmark Suite

The toolkit includes a comprehensive benchmark suite located in the `benchmarks/` directory:

- **`benchmark_metrics_100k.py`**: Benchmarks fairness metrics computation on 100k samples
- **`benchmark_pipeline.py`**: Benchmarks pipeline operations across different dataset sizes
- **`benchmark_bootstrap.py`**: Benchmarks bootstrap confidence interval computation

### Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/benchmark_metrics_100k.py
python benchmarks/benchmark_pipeline.py
python benchmarks/benchmark_bootstrap.py
```

### Typical Performance (Reference Hardware)

**Metrics Computation (100k samples):**
- Demographic Parity Difference: ~0.5-1.0 seconds
- Equalized Odds Difference: ~0.8-1.5 seconds
- MAE Parity Difference: ~0.6-1.2 seconds
- Intersectional analysis: ~2-4x slower (depends on number of groups)

**Pipeline Operations:**
- Bias detection: ~0.5-2.0 seconds (10k samples)
- Pipeline transformation: ~0.2-1.0 seconds (10k samples)
- Full pipeline (detect + transform): ~1-3 seconds (10k samples)

**Bootstrap Confidence Intervals:**
- Percentile method (1000 samples): ~5-15 seconds
- BCa method (1000 samples): ~10-30 seconds
- Performance scales linearly with number of bootstrap samples

*Note: Actual performance depends on hardware, dataset characteristics, and Python version.*

---

## Performance Characteristics

### Computational Complexity

**Fairness Metrics:**
- **Time Complexity**: O(n) where n is the number of samples
- **Space Complexity**: O(n) for storing predictions and sensitive attributes
- **Bootstrap CI**: O(n × B) where B is the number of bootstrap samples

**Pipeline Operations:**
- **Bias Detection**: O(n × m) where m is the number of features
- **Transformations**: O(n × m) for most transformers
- **Proxy Detection**: O(m²) for correlation computation

**Intersectional Analysis:**
- **Time Complexity**: O(n × g) where g is the number of intersectional groups
- **Space Complexity**: O(g) for storing group statistics
- Can be significantly slower when many groups are present

### Bottlenecks

1. **Bootstrap Confidence Intervals**: The most computationally expensive operation
   - Use `ci_method="percentile"` for faster computation
   - Reduce `ci_samples` for quicker results (at cost of accuracy)
   - Consider disabling CI for quick checks: `with_ci=False`

2. **Intersectional Analysis**: Slower due to increased number of groups
   - Use single-attribute analysis when possible
   - Filter to most important intersectional groups if needed

3. **Large Datasets**: Memory and computation time increase linearly
   - Use batch processing for very large datasets
   - Consider sampling for exploratory analysis

---

## Optimization Tips

### 1. Disable Confidence Intervals for Quick Checks

```python
# Fast check without CI
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=False  # Skip bootstrap CI computation
)
```

### 2. Use Percentile Method for Bootstrap CI

```python
# Faster CI method
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True,
    ci_method="percentile",  # Faster than "bca"
    ci_samples=500  # Fewer samples = faster
)
```

### 3. Reduce Minimum Group Size (When Appropriate)

```python
# Lower threshold for faster computation (use with caution)
analyzer = FairnessAnalyzer(min_group_size=20)  # Default is 30
```

### 4. Batch Processing for Large Datasets

```python
# Process in batches
batch_size = 10_000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    result = analyzer.demographic_parity_difference(
        y_pred=batch["y_pred"].to_numpy(),
        sensitive=batch["gender"].to_numpy(),
        with_ci=False  # Disable CI for batch processing
    )
```

### 5. Cache Results When Possible

```python
# Cache metric results if computing multiple times
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_metric_cached(y_pred_hash, sensitive_hash):
    # Compute metric
    return analyzer.demographic_parity_difference(...)
```

### 6. Use Native Backend (Fastest)

```python
# Native backend is typically fastest
analyzer = FairnessAnalyzer(backend="native")
```

### 7. Parallel Processing (Advanced)

For very large datasets, consider parallel processing:

```python
from multiprocessing import Pool

def compute_metric_for_group(args):
    group_name, group_data = args
    return analyzer.demographic_parity_difference(
        y_pred=group_data["y_pred"],
        sensitive=group_data["sensitive"]
    )

# Process groups in parallel
with Pool() as pool:
    results = pool.map(compute_metric_for_group, group_data_list)
```

---

## Scalability Considerations

### Dataset Size Guidelines

| Dataset Size | Recommended Approach | Notes |
|-------------|---------------------|-------|
| < 10k samples | Full analysis with CI | Fast enough for interactive use |
| 10k - 100k samples | Full analysis, consider reducing CI samples | Good balance of speed and accuracy |
| 100k - 1M samples | Batch processing or sampling | Use `with_ci=False` for quick checks |
| > 1M samples | Sampling or distributed processing | Consider using Spark/Dask for very large datasets |

### Group Size Considerations

- **Minimum Group Size**: Larger `min_group_size` values reduce computation time but may exclude important groups
- **Number of Groups**: More groups (especially in intersectional analysis) increase computation time
- **Group Imbalance**: Highly imbalanced groups may require more bootstrap samples for accurate CI

### Memory Usage

**Typical Memory Footprint:**
- Base toolkit: ~50-100 MB
- Per 100k samples: ~10-20 MB (depending on data types)
- Bootstrap CI (1000 samples): ~50-100 MB additional memory

**Memory Optimization:**
- Use `with_ci=False` to reduce memory usage
- Process data in batches for very large datasets
- Use appropriate data types (e.g., `int8` instead of `int64` when possible)

---

## Memory Usage

### Memory Profiling

```python
# Profile memory usage
import tracemalloc

tracemalloc.start()
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True
)
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()
```

### Memory-Efficient Patterns

1. **Use Generators for Large Datasets**:
   ```python
   def process_in_chunks(df, chunk_size=10000):
       for i in range(0, len(df), chunk_size):
           yield df.iloc[i:i + chunk_size]
   ```

2. **Clear Intermediate Results**:
   ```python
   result = analyzer.demographic_parity_difference(...)
   # Process result
   del result  # Explicitly free memory if needed
   ```

3. **Use Sparse Data Structures** (when applicable):
   ```python
   from scipy.sparse import csr_matrix
   # Use sparse matrices for very sparse data
   ```

---

## CI/CD Integration

### Performance Regression Testing

The toolkit includes performance benchmarks that can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/ci.yml
- name: Run performance benchmarks
  run: |
    python benchmarks/benchmark_metrics_100k.py > benchmark_metrics.txt
    python benchmarks/benchmark_pipeline.py > benchmark_pipeline.txt
    
    # Check for performance regressions
    python -c "
    import re
    with open('benchmark_metrics.txt') as f:
        content = f.read()
        # Extract timing information and check thresholds
        # Fail if performance degrades significantly
    "
```

### Performance Monitoring

Track performance over time:

```python
# Log performance metrics
import time
import json

start = time.time()
result = analyzer.demographic_parity_difference(...)
duration = time.time() - start

performance_log = {
    "metric": "demographic_parity_difference",
    "duration": duration,
    "n_samples": len(y_pred),
    "timestamp": time.time()
}

# Save to file or send to monitoring system
with open("performance_log.json", "a") as f:
    f.write(json.dumps(performance_log) + "\n")
```

### Benchmark Baselines

Establish performance baselines for your use case:

```bash
# Run benchmarks and save results
python benchmarks/benchmark_metrics_100k.py > baseline_metrics.txt
python benchmarks/benchmark_pipeline.py > baseline_pipeline.txt

# Compare against baselines in CI
python -c "
# Load baseline and current results
# Compare and fail if performance degrades > 20%
"
```

---

## Best Practices

1. **Start with Quick Checks**: Use `with_ci=False` for initial exploration
2. **Enable CI for Production**: Always use confidence intervals for production validation
3. **Profile Before Optimizing**: Use profiling tools to identify actual bottlenecks
4. **Monitor Performance**: Track performance metrics over time
5. **Set Appropriate Thresholds**: Balance accuracy (more CI samples) vs speed
6. **Use Appropriate Backend**: Native backend is typically fastest unless you need adapter features

---

## Troubleshooting Performance Issues

### Slow Metric Computation

**Symptoms**: Metrics take > 10 seconds for 100k samples

**Solutions**:
- Check if CI is enabled (disable for quick checks)
- Reduce `ci_samples` if CI is needed
- Verify you're using native backend
- Check for memory pressure (swap usage)

### High Memory Usage

**Symptoms**: Memory usage > 500 MB for 100k samples

**Solutions**:
- Process data in batches
- Disable CI if not needed
- Check for memory leaks in custom code
- Use appropriate data types

### Slow Pipeline Operations

**Symptoms**: Pipeline operations take > 5 seconds for 10k samples

**Solutions**:
- Check number of pipeline steps
- Verify transformer implementations are efficient
- Consider caching transformer fits
- Profile individual steps to identify bottlenecks

---

## Additional Resources

- **Benchmark Suite**: See `benchmarks/README.md` for detailed benchmark documentation
- **API Reference**: See `docs/api.md` for complete API documentation
- **Integration Guide**: See `docs/integration_guide.md` for integration examples

---

## Performance Comparison

### Backend Performance

| Backend | Speed | Features | Dependencies |
|---------|-------|----------|---------------|
| Native | Fastest | Core metrics | None (always available) |
| Fairlearn | Medium | Additional metrics | fairlearn |
| Aequitas | Slower | Comprehensive reports | aequitas |

*Recommendation: Use native backend unless you need specific adapter features.*

### CI Method Performance

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Percentile | Fast | Good | General use |
| BCa | Slower | Better | When accuracy is critical |

*Recommendation: Use percentile for most cases, BCa when accuracy is critical.*

---

For questions or performance issues, see the [Integration Guide](integration_guide.md) or open an issue on GitHub.
