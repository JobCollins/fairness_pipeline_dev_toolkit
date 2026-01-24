# Integration Guide

This guide demonstrates how to integrate the Fairness Pipeline Development Toolkit into various ML workflows, CI/CD pipelines, and production systems.

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Integration Patterns](#integration-patterns)
  - [Standalone Fairness Validation](#standalone-fairness-validation)
  - [Embedding in ML Pipelines](#embedding-in-ml-pipelines)
  - [CI/CD Integration](#cicd-integration)
  - [Production Monitoring](#production-monitoring)
  - [MLflow Integration](#mlflow-integration)
- [Configuration Management](#configuration-management)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Basic Installation

```bash
pip install fairness-pipeline-dev-toolkit
```

### With Optional Dependencies

For specific use cases, install additional features:

```bash
# For fairness-aware training
pip install fairness-pipeline-dev-toolkit[training]

# For production monitoring
pip install fairness-pipeline-dev-toolkit[monitoring]

# For external metric backends
pip install fairness-pipeline-dev-toolkit[adapters]

# All features
pip install fairness-pipeline-dev-toolkit[training,monitoring,adapters]
```

### Verify Installation

```python
from fairness_pipeline_dev_toolkit import __version__
print(__version__)  # Should print "0.5.0"

# Test CLI
import subprocess
subprocess.run(["fairpipe", "version"], check=True)
```

---

## Integration Patterns

### Standalone Fairness Validation

Use this pattern when you want to validate fairness of existing model predictions without modifying your training pipeline.

#### Use Case: Post-Training Validation

```python
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

# Load predictions
df = pd.read_csv("predictions.csv")

# Initialize analyzer
analyzer = FairnessAnalyzer(min_group_size=30)

# Compute fairness metrics
result = analyzer.demographic_parity_difference(
    y_pred=df["y_pred"].to_numpy(),
    sensitive=df["gender"].to_numpy(),
    with_ci=True,
    ci_level=0.95
)

# Check if fairness threshold is met
threshold = 0.05
if result.value <= threshold:
    print(f"✅ Fairness check passed: DPD = {result.value:.4f}")
else:
    print(f"❌ Fairness check failed: DPD = {result.value:.4f} (threshold: {threshold})")
    print(f"95% CI: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
```

#### Use Case: Batch Validation Script

```python
#!/usr/bin/env python3
"""Validate fairness for multiple models."""

import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
import sys

def validate_model(predictions_path: str, threshold: float = 0.05) -> bool:
    """Validate fairness for a single model."""
    df = pd.read_csv(predictions_path)
    analyzer = FairnessAnalyzer(min_group_size=30)
    
    result = analyzer.demographic_parity_difference(
        y_pred=df["y_pred"].to_numpy(),
        sensitive=df["gender"].to_numpy(),
        with_ci=True
    )
    
    passed = result.value <= threshold
    print(f"Model: {predictions_path}")
    print(f"DPD: {result.value:.4f} (threshold: {threshold})")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
    print()
    
    return passed

if __name__ == "__main__":
    predictions_files = sys.argv[1:]
    all_passed = all(validate_model(f) for f in predictions_files)
    sys.exit(0 if all_passed else 1)
```

---

### Embedding in ML Pipelines

Use this pattern when you want to integrate fairness directly into your training and inference pipelines.

#### Use Case: scikit-learn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fairness_pipeline_dev_toolkit.pipeline import InstanceReweighting
from fairness_pipeline_dev_toolkit.training import ReductionsWrapper

# Option 1: Add fairness transformer to preprocessing
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reweigh', InstanceReweighting(sensitive="gender")),
])

# Option 2: Use fairness-aware training wrapper
fair_model = ReductionsWrapper(
    LogisticRegression(),
    constraint="demographic_parity",
    eps=0.01
)

# Full pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
A_train = X_train["gender"]  # Sensitive attribute

# Preprocess
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_test_transformed = preprocessing_pipeline.transform(X_test)

# Train with fairness constraints
fair_model.fit(X_train_transformed, y_train, sensitive_features=A_train)

# Predict
y_pred = fair_model.predict(X_test_transformed)

# Validate
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
analyzer = FairnessAnalyzer()
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=X_test["gender"].to_numpy()
)
print(f"Final DPD: {result.value:.4f}")
```

#### Use Case: PyTorch Training Integration

```python
import torch
import torch.nn as nn
from fairness_pipeline_dev_toolkit.training import (
    FairnessRegularizerLoss,
    LagrangianFairnessTrainer
)

# Define model
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = MyModel(input_dim=10)

# Option 1: Use fairness regularizer in loss
criterion = FairnessRegularizerLoss(
    base_loss=nn.BCELoss(),
    eta=0.5,
    sensitive_attribute=sensitive_train
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

# Option 2: Use Lagrangian trainer
trainer = LagrangianFairnessTrainer(
    model=model,
    fairness="demographic_parity",
    dp_tol=0.02
)
trainer.train(X_train, y_train, sensitive_train)
```

#### Use Case: Custom Training Loop

```python
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.pipeline import build_pipeline, apply_pipeline, load_config

# Load configuration
config = load_config("pipeline.config.yml")

# Build and apply preprocessing pipeline
pipeline = build_pipeline(config)
X_transformed, metadata = apply_pipeline(pipeline, X)

# Train your model (any framework)
model = train_model(X_transformed, y)

# Validate fairness
analyzer = FairnessAnalyzer()
y_pred = model.predict(X_test_transformed)
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=X_test["gender"].to_numpy(),
    with_ci=True
)

# Log results
print(f"Fairness metric: {result.value:.4f}")
print(f"95% CI: {result.ci}")
```

---

### CI/CD Integration

Use this pattern to gate deployments based on fairness metrics.

#### Use Case: GitHub Actions

```yaml
# .github/workflows/fairness-check.yml
name: Fairness Validation

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  fairness-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install toolkit
        run: pip install fairness-pipeline-dev-toolkit
      
      - name: Run fairness validation
        run: |
          fairpipe validate \
            --csv test_predictions.csv \
            --y-true y_true \
            --y-pred y_pred \
            --sensitive gender \
            --min-group-size 30 \
            --with-ci \
            --out artifacts/fairness_report.md
      
      - name: Check fairness threshold
        run: |
          python -c "
          import pandas as pd
          from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
          
          df = pd.read_csv('test_predictions.csv')
          analyzer = FairnessAnalyzer(min_group_size=30)
          result = analyzer.demographic_parity_difference(
              y_pred=df['y_pred'].to_numpy(),
              sensitive=df['gender'].to_numpy()
          )
          
          threshold = 0.05
          if result.value > threshold:
              print(f'❌ Fairness check failed: DPD = {result.value:.4f} > {threshold}')
              exit(1)
          else:
              print(f'✅ Fairness check passed: DPD = {result.value:.4f}')
          "
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: fairness-report
          path: artifacts/fairness_report.md
```

#### Use Case: Performance Benchmarking in CI/CD

Add performance regression testing to your CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  performance-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      
      - name: Run performance benchmarks
        run: |
          echo "Running performance benchmarks..."
          python benchmarks/benchmark_metrics_100k.py > benchmark_metrics.txt 2>&1
          python benchmarks/benchmark_pipeline.py > benchmark_pipeline.txt 2>&1
          python benchmarks/benchmark_bootstrap.py > benchmark_bootstrap.txt 2>&1
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: |
            benchmark_*.txt
      
      - name: Check for performance regressions
        run: |
          # Extract timing information and compare against baselines
          # Fail if performance degrades significantly (>20%)
          python -c "
          import re
          import sys
          
          # Example: Check if metrics benchmark exceeds threshold
          with open('benchmark_metrics.txt') as f:
              content = f.read()
              # Add your performance regression checks here
              print('Performance benchmarks completed')
          "
```

#### Use Case: Automated Release Workflow

Automate releases with GitHub Actions:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'  # Trigger on version tags

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write  # For PyPI trusted publishing
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install build dependencies
        run: |
          pip install build twine
      
      - name: Extract version from tag
        id: version
        run: |
          VERSION=${GITHUB_REF#refs/tags/v}
          echo "version=$VERSION" >> $GITHUB_OUTPUT
      
      - name: Build distribution packages
        run: python -m build
      
      - name: Check package
        run: twine check dist/*
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ steps.version.outputs.version }}
          release_name: Release v${{ steps.version.outputs.version }}
          body: |
            Release v${{ steps.version.outputs.version }}
            
            See CHANGELOG.md for details.
          draft: false
          prerelease: false
```

#### Use Case: pytest Integration

```python
# tests/test_fairness.py
import pytest
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.integration import assert_fairness

def test_model_fairness():
    """Test that model predictions meet fairness threshold."""
    # Load test data
    df = pd.read_csv("test_predictions.csv")
    
    # Compute fairness metric
    analyzer = FairnessAnalyzer(min_group_size=30)
    result = analyzer.demographic_parity_difference(
        y_pred=df["y_pred"].to_numpy(),
        sensitive=df["gender"].to_numpy(),
        with_ci=True
    )
    
    # Assert fairness threshold
    assert_fairness(
        value=result.value,
        threshold=0.05,
        context="Demographic parity difference"
    )
    
    # Additional checks
    assert result.ci is not None, "Confidence interval should be computed"
    assert result.ci[1] < 0.10, "Upper CI should be below 10%"

@pytest.mark.parametrize("sensitive_attr", ["gender", "race"])
def test_multiple_attributes(sensitive_attr):
    """Test fairness across multiple sensitive attributes."""
    df = pd.read_csv("test_predictions.csv")
    analyzer = FairnessAnalyzer(min_group_size=30)
    
    result = analyzer.demographic_parity_difference(
        y_pred=df["y_pred"].to_numpy(),
        sensitive=df[sensitive_attr].to_numpy(),
        with_ci=True
    )
    
    assert_fairness(result.value, threshold=0.05, context=sensitive_attr)
```

Run tests:
```bash
pytest tests/test_fairness.py -v
```

#### Use Case: Pre-commit Hook

```python
# .pre-commit-hooks/fairness-check.py
#!/usr/bin/env python3
"""Pre-commit hook to check fairness of model predictions."""

import sys
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

def main():
    predictions_path = "predictions.csv"
    
    try:
        df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"⚠️  {predictions_path} not found, skipping fairness check")
        return 0
    
    analyzer = FairnessAnalyzer(min_group_size=30)
    result = analyzer.demographic_parity_difference(
        y_pred=df["y_pred"].to_numpy(),
        sensitive=df["gender"].to_numpy()
    )
    
    threshold = 0.05
    if result.value > threshold:
        print(f"❌ Fairness check failed: DPD = {result.value:.4f} > {threshold}")
        return 1
    
    print(f"✅ Fairness check passed: DPD = {result.value:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

### Production Monitoring

Use this pattern to monitor fairness metrics in production systems.

#### Use Case: Real-Time Tracking

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    RealTimeFairnessTracker,
    TrackerConfig,
    ColumnMap
)
import pandas as pd

# Initialize tracker
tracker = RealTimeFairnessTracker(
    TrackerConfig(
        window_size=10_000,
        min_group_size=30
    ),
    artifacts_dir="artifacts/monitoring"
)

# Define column mapping
column_map = ColumnMap(
    y_true="y_true",
    y_pred="y_pred",
    sensitive="gender"
)

# Process batches of predictions
def process_predictions_batch(batch_df: pd.DataFrame):
    """Process a batch of predictions and update fairness metrics."""
    tracker.process_batch(batch_df, column_map)
    
    # Get current metrics
    metrics = tracker.get_current_metrics()
    print(f"Current DPD: {metrics.get('demographic_parity_difference', 'N/A')}")

# In your inference service
while True:
    batch = get_next_batch()  # Your function to get predictions
    process_predictions_batch(batch)
```

#### Use Case: Drift Detection

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    FairnessDriftAndAlertEngine,
    DriftConfig
)

# Initialize drift detector
engine = FairnessDriftAndAlertEngine(
    DriftConfig(
        ks_threshold=0.05,
        alert_on_drift=True
    )
)

# Set reference metrics (from training/validation)
reference_metrics = {
    "demographic_parity_difference": 0.03,
    "equalized_odds_difference": 0.02
}

# Check for drift periodically
def check_drift(current_metrics: dict):
    """Check if current metrics have drifted from reference."""
    alerts = engine.check_drift(reference_metrics, current_metrics)
    
    if alerts:
        for alert in alerts:
            print(f"⚠️  Alert: {alert.message}")
            # Send to alerting system (e.g., PagerDuty, Slack)
            send_alert(alert)
    
    return alerts

# In monitoring loop
current_metrics = compute_current_metrics()  # Your function
alerts = check_drift(current_metrics)
```

#### Use Case: Dashboard Integration

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    FairnessReportingDashboard,
    ReportConfig
)

# Initialize dashboard
dashboard = FairnessReportingDashboard(
    ReportConfig(
        metrics_dir="artifacts/monitoring",
        update_interval=3600  # Update every hour
    )
)

# Generate report
dashboard.generate_report(output_path="artifacts/fairness_dashboard.html")

# Or serve as web app (if using Streamlit/Dash)
# See apps/monitoring_streamlit_app.py for example
```

---

### MLflow Integration

Use this pattern to log fairness metrics alongside model metrics in MLflow.

#### Use Case: Logging Workflow Results

```python
from fairness_pipeline_dev_toolkit.integration import (
    execute_workflow,
    log_workflow_results
)
from fairness_pipeline_dev_toolkit.pipeline import load_config
import pandas as pd

# Load configuration and data
config = load_config("config.yml")
df = pd.read_csv("data.csv")

# Execute workflow
result = execute_workflow(
    config=config,
    df=df,
    output_dir="artifacts/workflow",
    min_group_size=30,
    mlflow_experiment="fairness_workflow",
    mlflow_run_name="run_001"
)

# Log results to MLflow (if not already logged)
log_workflow_results(
    result=result,
    experiment_name="fairness_workflow",
    run_name="run_001"
)
```

#### Use Case: Custom MLflow Logging

```python
import mlflow
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.integration import log_fairness_metrics

# Start MLflow run
mlflow.start_run(run_name="fairness_evaluation")

# Compute fairness metrics
analyzer = FairnessAnalyzer(min_group_size=30)
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=gender,
    with_ci=True
)

# Log to MLflow
log_fairness_metrics(
    metrics={
        "demographic_parity_difference": result.value,
        "dpd_ci_lower": result.ci[0],
        "dpd_ci_upper": result.ci[1],
    },
    experiment_name="fairness_workflow"
)

mlflow.end_run()
```

---

## Configuration Management

### Configuration File Discovery

The toolkit supports automatic configuration file discovery:

```python
from fairness_pipeline_dev_toolkit.pipeline.config import find_config_file, load_config

# Find config file (checks environment variables and default locations)
config_path = find_config_file("config.yml")
if config_path:
    config = load_config(config_path)
else:
    # Use default or raise error
    raise FileNotFoundError("Configuration file not found")
```

### Environment-Based Configuration

```python
import os
from fairness_pipeline_dev_toolkit.pipeline.config import load_config

# Set configuration path via environment variable
os.environ["FAIRPIPE_CONFIG_PATH"] = "production_config.yml"
config = load_config(os.environ["FAIRPIPE_CONFIG_PATH"])
```

### Programmatic Configuration

```python
from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig, PipelineStep

# Create configuration programmatically
config = PipelineConfig(
    sensitive=["gender", "race"],
    pipeline=[
        PipelineStep(
            name="reweigh",
            transformer="InstanceReweighting"
        ),
        PipelineStep(
            name="repair",
            transformer="DisparateImpactRemover",
            params={
                "features": ["score"],
                "sensitive": "gender",
                "repair_level": 0.8
            }
        )
    ]
)
```

---

## Environment Variables

The toolkit supports configuration via environment variables:

```bash
# Configuration file path
export FAIRPIPE_CONFIG_PATH="config.yml"

# Minimum group size
export FAIRPIPE_MIN_GROUP_SIZE=30

# MLflow experiment name
export FAIRPIPE_MLFLOW_EXPERIMENT="fairness_workflow"
```

Access in Python:

```python
from fairness_pipeline_dev_toolkit.config.env import (
    FAIRPIPE_CONFIG_PATH,
    FAIRPIPE_MIN_GROUP_SIZE,
    FAIRPIPE_MLFLOW_EXPERIMENT,
    get_env_int
)

# Get environment variables
config_path = os.getenv(FAIRPIPE_CONFIG_PATH, "config.yml")
min_group_size = get_env_int(FAIRPIPE_MIN_GROUP_SIZE, default=30)
```

---

## Best Practices

### 1. Minimum Group Size

Always set an appropriate minimum group size based on your data:

```python
# For small datasets
analyzer = FairnessAnalyzer(min_group_size=10)

# For large datasets (recommended)
analyzer = FairnessAnalyzer(min_group_size=30)
```

### 2. Confidence Intervals

Always compute confidence intervals for production use:

```python
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True,
    ci_level=0.95,
    ci_samples=1000  # More samples = more accurate CI
)
```

### 3. Multiple Sensitive Attributes

Test fairness across all relevant sensitive attributes:

```python
sensitive_attrs = ["gender", "race", "age_group"]

for attr in sensitive_attrs:
    result = analyzer.demographic_parity_difference(
        y_pred=y_pred,
        sensitive=df[attr].to_numpy(),
        with_ci=True
    )
    print(f"{attr}: DPD = {result.value:.4f}")
```

### 4. Intersectional Analysis

For intersectional fairness, use multiple attributes:

```python
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=None,  # Not used when intersectional=True
    intersectional=True,
    attrs_df=df[["gender", "race"]],
    columns=["gender", "race"],
    with_ci=True
)
```

### 5. Error Handling

Always handle edge cases:

```python
from fairness_pipeline_dev_toolkit.exceptions import (
    MetricComputationError,
    ConfigValidationError
)

try:
    result = analyzer.demographic_parity_difference(
        y_pred=y_pred,
        sensitive=sensitive,
        with_ci=True
    )
    
    if result.value is None or np.isnan(result.value):
        print("⚠️  Insufficient data for fairness computation")
    else:
        print(f"DPD: {result.value:.4f}")
        
except MetricComputationError as e:
    print(f"Error computing metric: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 6. Logging and Monitoring

Log fairness metrics for tracking over time:

```python
import logging
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analyzer = FairnessAnalyzer(min_group_size=30)
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True
)

logger.info(
    f"Fairness metric computed: DPD={result.value:.4f}, "
    f"CI=[{result.ci[0]:.4f}, {result.ci[1]:.4f}]"
)
```

---

## Troubleshooting

### Common Issues

#### 1. "Insufficient data" or NaN results

**Problem:** Metric returns NaN or `None`

**Solutions:**
- Reduce `min_group_size` (but be cautious with small groups)
- Check data quality (missing values, incorrect group labels)
- Use `nan_policy="include"` if you want to include NaN values

```python
analyzer = FairnessAnalyzer(
    min_group_size=10,  # Lower threshold
    nan_policy="include"  # Include NaN values
)
```

#### 2. Backend not available

**Problem:** `RuntimeError: Requested backend 'fairlearn' is not available`

**Solutions:**
- Install optional dependencies: `pip install fairness-pipeline-dev-toolkit[adapters]`
- Use `backend="native"` (always available)
- Let toolkit auto-select: `backend=None`

```python
analyzer = FairnessAnalyzer(backend="native")  # Always works
```

#### 3. Configuration file not found

**Problem:** `FileNotFoundError` when loading config

**Solutions:**
- Use absolute paths
- Set `FAIRPIPE_CONFIG_PATH` environment variable
- Use `find_config_file()` to locate config

```python
from fairness_pipeline_dev_toolkit.pipeline.config import find_config_file

config_path = find_config_file("config.yml")
if config_path:
    config = load_config(config_path)
```

#### 4. Slow bootstrap CI computation

**Problem:** Computing confidence intervals takes too long

**Solutions:**
- Reduce `ci_samples` (default: 1000)
- Use `ci_method="percentile"` instead of `"bca"` (faster)
- Disable CI for quick checks: `with_ci=False`

```python
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True,
    ci_samples=500,  # Fewer samples = faster
    ci_method="percentile"  # Faster method
)
```

### Getting Help

- **Documentation:** See [API Reference](api.md) for detailed API documentation
- **Examples:** Check `demo_*.ipynb` notebooks in the repository
- **Issues:** Report issues on [GitHub](https://github.com/SvrusIO/fAIr/issues)

---

## Next Steps

- Read the [API Reference](api.md) for complete API documentation
- Explore the demo notebooks (`demo_*.ipynb`) for more examples
- Review the [README](../README.md) for quick start guide
- Check [ARCHITECTURE.md](ADR-001-architecture.md) for design decisions
