# Fairness Pipeline Development Toolkit - Comprehensive User Guide

**Version:** 0.6.0

This guide walks you through using the Fairness Pipeline Development Toolkit across the complete model development cycle—from initial data exploration to production monitoring. Whether you're building a new model or improving an existing one, this toolkit provides the tools and workflows to ensure fairness at every stage.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Model Development Cycle Overview](#model-development-cycle-overview)
4. [Phase 1: Data Preparation & Exploration](#phase-1-data-preparation--exploration)
5. [Phase 2: Baseline Fairness Measurement](#phase-2-baseline-fairness-measurement)
6. [Phase 3: Bias Detection & Mitigation](#phase-3-bias-detection--mitigation)
7. [Phase 4: Fairness-Aware Model Training](#phase-4-fairness-aware-model-training)
8. [Phase 5: Validation & Testing](#phase-5-validation--testing)
9. [Phase 6: Deployment Preparation](#phase-6-deployment-preparation)
10. [Phase 7: Production Monitoring](#phase-7-production-monitoring)
11. [Integrated Workflow (Quick Path)](#integrated-workflow-quick-path)
12. [Best Practices & Troubleshooting](#best-practices--troubleshooting)
13. [Advanced Topics](#advanced-topics)

---

## Introduction

The Fairness Pipeline Development Toolkit is a unified, statistically-rigorous framework for **detecting**, **mitigating**, **training**, and **validating** fairness in ML workflows. It provides:

- **Modular Components**: Use individual modules (Measurement, Pipeline, Training, Monitoring) as needed
- **Integrated Workflow**: End-to-end three-step process from data to validated model
- **Statistical Rigor**: Bootstrap confidence intervals, effect sizes, and hypothesis testing
- **CI/CD Ready**: Automated validation, MLflow integration, and pytest plugins
- **Production Monitoring**: Real-time tracking, drift detection, and alerting

### Key Concepts

- **Sensitive Attributes**: Protected characteristics (e.g., gender, race, age) that should not lead to unfair treatment
- **Fairness Metrics**: Quantitative measures of fairness (demographic parity, equalized odds, etc.)
- **Bias Mitigation**: Techniques to reduce unfairness in data and models
- **Fairness Constraints**: Requirements that models must satisfy (e.g., demographic parity within 5%)

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Basic Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core toolkit
pip install -e .[adapters]

# Optional: Install training and monitoring extras
pip install -e .[training,monitoring]
```

### Installation with Pinned Dependencies

For reproducible environments:

```bash
pip install pip-tools
pip-compile --extra training --extra monitoring --extra adapters \
    --output-file=requirements.txt requirements-dev.in
pip install -r requirements.txt
```

> **Note**: PyTorch wheels depend on platform/accelerator support. Follow the commands from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) before enabling the `training` extra.

### Verify Installation

```bash
fairpipe version
```

You should see the version number (e.g., `0.6.0`).

---

## Model Development Cycle Overview

The toolkit supports fairness work at every stage of the ML lifecycle:

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL DEVELOPMENT CYCLE                    │
└─────────────────────────────────────────────────────────────┘

Phase 1: Data Preparation & Exploration
  ↓
Phase 2: Baseline Fairness Measurement
  ↓
Phase 3: Bias Detection & Mitigation
  ↓
Phase 4: Fairness-Aware Model Training
  ↓
Phase 5: Validation & Testing
  ↓
Phase 6: Deployment Preparation
  ↓
Phase 7: Production Monitoring
```

Each phase has specific tools and workflows. You can:
- **Use individual modules** for specific tasks
- **Use the integrated workflow** for end-to-end automation
- **Combine approaches** as needed for your use case

---

## Phase 1: Data Preparation & Exploration

### Objectives

- Identify sensitive attributes in your dataset
- Understand data distributions and potential bias sources
- Prepare data for fairness analysis

### Step 1: Load and Inspect Data

```python
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("your_data.csv")

# Inspect structure
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
print(df.describe())
```

### Step 2: Identify Sensitive Attributes

Sensitive attributes are protected characteristics that should not lead to unfair treatment. Common examples:
- Gender, race, ethnicity
- Age (especially for age-protected groups)
- Disability status
- Religion, national origin

```python
# Example: Identify sensitive attributes
sensitive_attributes = ["gender", "race"]  # Adjust to your dataset

# Check distributions
for attr in sensitive_attributes:
    print(f"\n{attr} distribution:")
    print(df[attr].value_counts(normalize=True))
```

### Step 3: Check Data Quality

```python
# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check for sufficient group sizes
for attr in sensitive_attributes:
    print(f"\n{attr} group sizes:")
    print(df[attr].value_counts())
```

**Minimum Group Size**: Ensure each group has at least 30 samples for statistical reliability. Smaller groups may need special handling.

### Step 4: Prepare Target Variable

```python
# Identify target column
target_column = "y"  # or "label", "target", etc.

# Check target distribution
print(f"Target distribution:")
print(df[target_column].value_counts(normalize=True))

# Check target by sensitive attribute (initial bias check)
for attr in sensitive_attributes:
    print(f"\nTarget distribution by {attr}:")
    print(pd.crosstab(df[attr], df[target_column], normalize='index'))
```

### Output

At this stage, you should have:
- ✅ Identified sensitive attributes
- ✅ Confirmed sufficient sample sizes per group
- ✅ Understood data distributions
- ✅ Prepared target variable

**Next**: Move to Phase 2 to measure baseline fairness.

---

## Phase 2: Baseline Fairness Measurement

### Objectives

- Quantify existing fairness issues in raw data
- Establish baseline metrics for comparison
- Understand which groups are most affected

### Method 1: CLI Quick Check

The fastest way to measure baseline fairness:

```bash
fairpipe validate \
  --csv your_data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --with-ci \
  --ci-level 0.95 \
  --with-effects \
  --out baseline_report.md
```

### Method 2: Python API

For more control and integration:

```python
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

# Initialize analyzer
analyzer = FairnessAnalyzer(backend="native")  # or "fairlearn", "aequitas"

# Load data
df = pd.read_csv("your_data.csv")

# Measure baseline fairness
# Note: For baseline, you may not have predictions yet
# You can measure data-level fairness or use a simple baseline model

# If you have predictions:
baseline_results = analyzer.analyze(
    y_true=df["y_true"],
    y_pred=df["y_pred"],  # or baseline model predictions
    sensitive_features=df[["gender", "race"]],
    min_group_size=30
)

# View results
print("Baseline Fairness Metrics:")
for metric_name, result in baseline_results.items():
    print(f"\n{metric_name}:")
    print(f"  Value: {result.value:.4f}")
    if result.ci:
        print(f"  95% CI: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
    if result.effect_size:
        print(f"  Effect Size: {result.effect_size:.4f}")
```

### Key Metrics to Review

1. **Demographic Parity Difference**
   - Measures difference in selection rates between groups
   - Target: < 0.05 (5% difference)
   - Formula: `max(selection_rate) - min(selection_rate)`

2. **Equalized Odds Difference**
   - Measures difference in true positive rates (TPR) and false positive rates (FPR)
   - Target: < 0.10 (10% difference)
   - Important for classification tasks

3. **MAE Parity Difference** (for regression)
   - Measures difference in mean absolute error between groups
   - Target: < 0.10 relative difference

### Statistical Validation

The toolkit provides confidence intervals and effect sizes:

```python
# With confidence intervals
results = analyzer.analyze(
    y_true=df["y_true"],
    y_pred=df["y_pred"],
    sensitive_features=df[["gender"]],
    with_ci=True,
    ci_level=0.95,
    with_effects=True
)

# Results include:
# - Bootstrap confidence intervals
# - Effect sizes (risk ratio, Cohen's d)
# - Sample sizes per group
```

### Intersectional Analysis

Analyze fairness across combinations of sensitive attributes:

```python
# Analyze intersectional groups
results = analyzer.analyze(
    y_true=df["y_true"],
    y_pred=df["y_pred"],
    sensitive_features=df[["gender", "race"]],
    intersections=[["gender", "race"]],  # Analyze gender×race combinations
    min_group_size=30  # Minimum samples per intersectional group
)
```

### Output

At this stage, you should have:
- ✅ Baseline fairness metrics documented
- ✅ Confidence intervals for statistical rigor
- ✅ Understanding of which groups are most affected
- ✅ Baseline report saved for comparison

**Next**: Move to Phase 3 to detect and mitigate bias.

---

## Phase 3: Bias Detection & Mitigation

### Objectives

- Automatically detect bias in data
- Apply mitigation transformers to reduce bias
- Generate transformed dataset ready for training

### Step 1: Create Pipeline Configuration

Create a YAML configuration file (`pipeline.config.yml`):

```yaml
# Required: Sensitive attributes
sensitive: ["gender", "race"]

# Optional: Population benchmarks for representation checks
benchmarks:
  gender:
    M: 0.5
    F: 0.5
  race:
    A: 0.3
    B: 0.3
    C: 0.4

# Statistical test parameters
alpha: 0.05
proxy_threshold: 0.30  # Correlation threshold for proxy detection

# Pipeline: Bias mitigation transformers (applied in order)
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
  
  - name: repair
    transformer: "DisparateImpactRemover"
    params:
      features: ["score", "age", "education"]
      sensitive: "gender"
      repair_level: 0.8
  
  - name: drop_proxies
    transformer: "ProxyDropper"
    params:
      threshold: 0.30
      sensitive: ["gender", "race"]
```

### Available Transformers

1. **InstanceReweighting**
   - Reweights samples to balance group distributions
   - No parameters required

2. **DisparateImpactRemover**
   - Repairs features to reduce disparate impact
   - Parameters:
     - `features`: List of feature columns to repair
     - `sensitive`: Sensitive attribute to use
     - `repair_level`: Strength of repair (0.0 to 1.0)

3. **ReweighingTransformer**
   - Advanced reweighing with custom strategies
   - Parameters: `strategy` (e.g., "none", "demographic_parity")

4. **ProxyDropper**
   - Removes features that are proxies for sensitive attributes
   - Parameters:
     - `threshold`: Correlation threshold (default: 0.30)
     - `sensitive`: List of sensitive attributes

### Step 2: Run Bias Detection

The pipeline automatically detects bias before mitigation:

```bash
fairpipe pipeline \
  --config pipeline.config.yml \
  --csv your_data.csv \
  --out-csv artifacts/transformed_data.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_report.md
```

This will:
1. Run bias detectors (representation, statistical, proxy analysis)
2. Apply mitigation transformers
3. Generate transformed dataset
4. Save detection results and report

### Step 3: Review Detection Results

```python
import json

# Load detection results
with open("artifacts/detectors.json", "r") as f:
    detectors = json.load(f)

print("Bias Detection Results:")
for detector_name, results in detectors.items():
    print(f"\n{detector_name}:")
    print(f"  Issues Found: {results.get('issues_found', 0)}")
    if results.get('issues'):
        for issue in results['issues']:
            print(f"    - {issue}")
```

### Step 4: Inspect Transformed Data

```python
# Load original and transformed data
original = pd.read_csv("your_data.csv")
transformed = pd.read_csv("artifacts/transformed_data.csv")

# Compare distributions
print("Original data - Gender distribution:")
print(original["gender"].value_counts(normalize=True))

print("\nTransformed data - Gender distribution:")
print(transformed["gender"].value_counts(normalize=True))

# Compare feature statistics
print("\nFeature statistics comparison:")
print(original[["score", "age"]].describe())
print(transformed[["score", "age"]].describe())
```

### Output

At this stage, you should have:
- ✅ Bias detection report
- ✅ Transformed dataset with reduced bias
- ✅ Understanding of which mitigations were applied
- ✅ Pipeline configuration saved for reproducibility

**Next**: Move to Phase 4 to train a fairness-aware model.

---

## Phase 4: Fairness-Aware Model Training

### Objectives

- Train models with fairness constraints
- Explore fairness-accuracy trade-offs
- Select optimal model configuration

### Training Methods

The toolkit supports three training approaches:

1. **Reductions (scikit-learn)**: Constraint-based training using Fairlearn
2. **Regularized (PyTorch)**: Fairness penalties in loss function
3. **Lagrangian (PyTorch)**: Dual optimization with Lagrange multipliers

### Method 1: Reductions (scikit-learn) - Recommended for Most Cases

Best for: scikit-learn models, constraint-based fairness

```python
from fairness_pipeline_dev_toolkit.training import ReductionsWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load transformed data
df = pd.read_csv("artifacts/transformed_data.csv")

# Prepare features and target
X = df.drop(columns=["y", "gender", "race"])
y = df["y"]
A = df[["gender"]]  # Sensitive attributes

# Split data
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.2, random_state=42
)

# Create base estimator
base_estimator = LogisticRegression()

# Wrap with fairness constraints
fair_model = ReductionsWrapper(
    estimator=base_estimator,
    constraint="demographic_parity",  # or "equalized_odds"
    eps=0.01  # Constraint tolerance
)

# Train
fair_model.fit(X_train, y_train, sensitive_features=A_train)

# Predict
y_pred = fair_model.predict(X_test)
y_proba = fair_model.predict_proba(X_test)[:, 1]
```

### Method 2: Regularized (PyTorch)

Best for: Deep learning models, exploring fairness-accuracy trade-offs

```python
from fairness_pipeline_dev_toolkit.training.torch_.losses import FairnessRegularizerLoss
import torch
import torch.nn as nn

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# Prepare data
X_tensor = torch.FloatTensor(X_train.values)
y_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
A_tensor = torch.FloatTensor(A_train.values)

# Initialize model
model = SimpleNN(X_train.shape[1])

# Define loss with fairness regularizer
criterion = FairnessRegularizerLoss(
    base_loss=nn.BCELoss(),
    eta=0.5,  # Fairness regularization strength
    sensitive_features=A_tensor
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor, sensitive_features=A_tensor)
    loss.backward()
    optimizer.step()
```

### Method 3: Lagrangian (PyTorch)

Best for: Strict fairness constraints, dual optimization

```python
from fairness_pipeline_dev_toolkit.training import LagrangianFairnessTrainer

# Prepare data
train_data = {
    "X": X_train.values,
    "y": y_train.values,
    "sensitive": A_train["gender"].values
}

# Initialize trainer
trainer = LagrangianFairnessTrainer(
    model=model,
    fairness="demographic_parity",  # or "equal_opportunity"
    dp_tol=0.02,  # Demographic parity tolerance
    model_lr=0.001,
    lambda_lr=0.01,
    epochs=50,
    batch_size=128,
    device="cpu"  # or "cuda"
)

# Train
history = trainer.train(train_data)

# Access trained model
trained_model = trainer.model
```

### Exploring Fairness-Accuracy Trade-offs

Use the Pareto Frontier visualization to explore trade-offs:

```bash
fairpipe train-regularized \
  --csv artifacts/transformed_data.csv \
  --etas "0.0,0.2,0.5,1.0,2.0" \
  --epochs 50 \
  --lr 1e-3 \
  --out-json artifacts/pareto_points.json \
  --out-png artifacts/pareto.png
```

This generates a visualization showing the trade-off between accuracy and fairness for different regularization strengths.

### Post-Training Calibration

Calibrate probabilities to ensure fairness across groups:

```bash
fairpipe calibrate \
  --csv scores.csv \
  --method platt \
  --min-samples 20 \
  --out-csv artifacts/calibrated_scores.csv
```

Or in Python:

```python
from fairness_pipeline_dev_toolkit.training.postproc import GroupFairnessCalibrator

calibrator = GroupFairnessCalibrator(
    method="platt",  # or "isotonic"
    min_samples=20
)

calibrator.fit(y_true=y_train, y_score=y_proba_train, sensitive_features=A_train)
y_calibrated = calibrator.transform(y_score=y_proba_test, sensitive_features=A_test)
```

### Output

At this stage, you should have:
- ✅ Trained fairness-aware model
- ✅ Understanding of fairness-accuracy trade-offs
- ✅ Calibrated predictions (if needed)
- ✅ Training history and metrics

**Next**: Move to Phase 5 to validate the model.

---

## Phase 5: Validation & Testing

### Objectives

- Validate model meets fairness thresholds
- Compare to baseline metrics
- Generate validation reports
- Integrate into CI/CD pipelines

### Method 1: Integrated Workflow Validation

The integrated workflow automatically validates:

```bash
fairpipe run-pipeline \
  --config config.yml \
  --csv your_data.csv \
  --output-dir artifacts/ \
  --mlflow-experiment fairness_workflow \
  --min-group-size 30 \
  --train-size 0.8
```

This runs:
1. Baseline measurement
2. Transform + train
3. Final validation (compares to baseline and checks threshold)

### Method 2: Manual Validation

```python
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

# Measure final model fairness
analyzer = FairnessAnalyzer(backend="native")

final_results = analyzer.analyze(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test,
    min_group_size=30,
    with_ci=True,
    with_effects=True
)

# Compare to baseline
print("Baseline vs Final Metrics:")
baseline_dp = baseline_results["demographic_parity_difference"].value
final_dp = final_results["demographic_parity_difference"].value

print(f"Demographic Parity Difference:")
print(f"  Baseline: {baseline_dp:.4f}")
print(f"  Final:    {final_dp:.4f}")
print(f"  Improvement: {baseline_dp - final_dp:.4f}")

# Check threshold
threshold = 0.05
if abs(final_dp) <= threshold:
    print(f"✅ Validation PASSED: {final_dp:.4f} <= {threshold}")
else:
    print(f"❌ Validation FAILED: {final_dp:.4f} > {threshold}")
```

### Method 3: Pytest Integration

Use the pytest plugin for automated testing:

```python
# test_fairness.py
import pytest
from fairness_pipeline_dev_toolkit.integration.pytest_plugin import assert_fairness

def test_model_fairness():
    """Test that model meets fairness threshold."""
    assert_fairness(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test,
        metric="demographic_parity_difference",
        threshold=0.05,
        min_group_size=30
    )
```

Run tests:

```bash
pytest test_fairness.py -v
```

### Validation Report

Generate a comprehensive validation report:

```python
from fairness_pipeline_dev_toolkit.integration.reporting import generate_validation_report

report = generate_validation_report(
    baseline_metrics=baseline_results,
    final_metrics=final_results,
    validation_result=validation_result,
    output_path="artifacts/validation_report.md"
)
```

### MLflow Integration

Log results to MLflow for tracking:

```python
from fairness_pipeline_dev_toolkit.integration.mlflow_logger import log_workflow_results
import mlflow

mlflow.set_experiment("fairness_workflow")

with mlflow.start_run():
    log_workflow_results(
        baseline_metrics=baseline_results,
        final_metrics=final_results,
        validation_result=validation_result,
        model=model,
        config_path="config.yml"
    )
```

### Output

At this stage, you should have:
- ✅ Validation results (pass/fail)
- ✅ Comparison to baseline metrics
- ✅ Validation report
- ✅ MLflow tracking (if used)
- ✅ CI/CD integration (if used)

**Next**: Move to Phase 6 to prepare for deployment.

---

## Phase 6: Deployment Preparation

### Objectives

- Package model and artifacts
- Document fairness characteristics
- Prepare monitoring configuration
- Set up deployment checks

### Step 1: Save Model and Artifacts

```python
import joblib
import json
from pathlib import Path

# Create artifacts directory
artifacts_dir = Path("artifacts/deployment")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, artifacts_dir / "model.pkl")

# Save configuration
with open(artifacts_dir / "config.json", "w") as f:
    json.dump({
        "sensitive_attributes": ["gender", "race"],
        "fairness_metric": "demographic_parity_difference",
        "validation_threshold": 0.05,
        "min_group_size": 30,
        "model_type": "reductions",
        "constraint": "demographic_parity"
    }, f, indent=2)

# Save feature names
with open(artifacts_dir / "feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f, indent=2)

# Save validation results
with open(artifacts_dir / "validation_results.json", "w") as f:
    json.dump({
        "baseline_metrics": {k: v.value for k, v in baseline_results.items()},
        "final_metrics": {k: v.value for k, v in final_results.items()},
        "validation_passed": validation_result.passed
    }, f, indent=2)
```

### Step 2: Create Deployment Documentation

```markdown
# Model Deployment Documentation

## Model Information
- **Model Type**: ReductionsWrapper (Fairlearn)
- **Base Estimator**: LogisticRegression
- **Training Date**: 2024-01-15
- **Version**: 1.0.0

## Fairness Characteristics
- **Sensitive Attributes**: gender, race
- **Fairness Metric**: demographic_parity_difference
- **Validation Threshold**: 0.05
- **Final Metric Value**: 0.032
- **Validation Status**: ✅ PASSED

## Model Performance
- **Accuracy**: 0.85
- **AUC-ROC**: 0.91
- **Baseline Fairness**: 0.15 (demographic_parity_difference)
- **Final Fairness**: 0.032 (demographic_parity_difference)
- **Improvement**: 0.118

## Monitoring Requirements
- Track fairness metrics in production
- Alert if demographic_parity_difference > 0.05
- Minimum group size: 30 samples per group
```

### Step 3: Set Up Monitoring Configuration

Create monitoring configuration for production:

```python
# monitoring_config.py
from fairness_pipeline_dev_toolkit.monitoring import (
    TrackerConfig,
    DriftConfig,
    MonitoringSettings
)

monitoring_config = MonitoringSettings(
    tracker=TrackerConfig(
        window_size=10_000,  # Sliding window size
        min_group_size=30,   # Minimum samples per group
        metrics=["demographic_parity_difference", "equalized_odds_difference"]
    ),
    drift=DriftConfig(
        alpha=0.05,          # Significance level for drift detection
        threshold=0.05,     # Alert threshold for fairness metric
        use_wavelets=False  # Enable for multi-scale analysis
    ),
    alerts={
        "demographic_parity_difference": {
            "threshold": 0.05,
            "severity": "high"
        }
    }
)

# Save configuration
import json
with open("artifacts/deployment/monitoring_config.json", "w") as f:
    json.dump(monitoring_config.dict(), f, indent=2)
```

### Step 4: Create Deployment Checklist

```markdown
# Deployment Checklist

## Pre-Deployment
- [ ] Model validation passed
- [ ] Fairness thresholds met
- [ ] Model artifacts saved
- [ ] Configuration documented
- [ ] Monitoring setup configured
- [ ] Stakeholder approval obtained

## Deployment
- [ ] Model deployed to staging
- [ ] Monitoring active
- [ ] Alerts configured
- [ ] Dashboard accessible

## Post-Deployment
- [ ] Baseline metrics established
- [ ] Monitoring verified
- [ ] Team trained on monitoring dashboard
- [ ] Runbook created
```

### Output

At this stage, you should have:
- ✅ Model and artifacts packaged
- ✅ Deployment documentation
- ✅ Monitoring configuration
- ✅ Deployment checklist

**Next**: Move to Phase 7 for production monitoring.

---

## Phase 7: Production Monitoring

### Objectives

- Track fairness metrics in real-time
- Detect fairness drift
- Generate alerts for violations
- Create monitoring dashboards

### Step 1: Initialize Real-Time Tracker

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    RealTimeFairnessTracker,
    ColumnMap,
    TrackerConfig
)

# Initialize tracker
tracker = RealTimeFairnessTracker(
    config=TrackerConfig(
        window_size=10_000,  # Process last 10k samples
        min_group_size=30,   # Minimum samples per group
    ),
    artifacts_dir="artifacts/monitoring"
)

# Define column mapping
cmap = ColumnMap(
    y_pred="predictions",
    y_true="labels",  # Optional, if available
    protected=["gender", "race"],
    intersections=[["gender", "race"]]  # Optional intersectional analysis
)
```

### Step 2: Process Production Batches

```python
# In your production inference pipeline
def process_batch(predictions_df):
    """
    Process a batch of predictions and update fairness tracking.
    
    predictions_df should have columns:
    - predictions: model predictions
    - labels: true labels (if available)
    - gender, race: sensitive attributes
    """
    # Update tracker
    tracker.process_batch(predictions_df, cmap)
    
    # Get current metrics
    current_metrics = tracker.get_current_metrics()
    
    return current_metrics
```

### Step 3: Detect Drift and Generate Alerts

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    FairnessDriftAndAlertEngine,
    DriftConfig
)

# Initialize drift engine
drift_engine = FairnessDriftAndAlertEngine(
    config=DriftConfig(
        alpha=0.05,
        threshold=0.05,
        use_wavelets=False
    )
)

# Analyze for drift (call periodically, e.g., daily)
alerts = drift_engine.analyze(tracker.metrics_ts)

# Check for alerts
if alerts:
    print("⚠️ Fairness Alerts Detected:")
    for alert in alerts:
        print(f"  - {alert['metric']}: {alert['value']:.4f} "
              f"(threshold: {alert['threshold']:.4f})")
        print(f"    Severity: {alert['severity']}")
        print(f"    Message: {alert['message']}")
```

### Step 4: Generate Monitoring Reports

```python
from fairness_pipeline_dev_toolkit.monitoring import FairnessReportingDashboard

# Initialize dashboard
dashboard = FairnessReportingDashboard(
    tracker=tracker,
    artifacts_dir="artifacts/monitoring"
)

# Generate report
dashboard.generate_report(
    output_path="artifacts/monitoring/report.md",
    include_plots=True
)
```

### Step 5: Interactive Dashboards

Use the provided Streamlit or Dash apps:

**Streamlit:**
```bash
streamlit run apps/monitoring_streamlit_app.py
```

**Dash:**
```bash
python apps/monitoring_dash_app.py
```

### A/B Testing

Compare fairness between model versions:

```python
from fairness_pipeline_dev_toolkit.monitoring import FairnessABTestAnalyzer

# Compare two model versions
ab_analyzer = FairnessABTestAnalyzer()

results = ab_analyzer.compare(
    metrics_a=tracker_a.metrics_ts,
    metrics_b=tracker_b.metrics_ts,
    metric="demographic_parity_difference"
)

print(f"A/B Test Results:")
print(f"  Model A: {results['mean_a']:.4f}")
print(f"  Model B: {results['mean_b']:.4f}")
print(f"  Difference: {results['difference']:.4f}")
print(f"  P-value: {results['p_value']:.4f}")
print(f"  Significant: {results['significant']}")
```

### Output

At this stage, you should have:
- ✅ Real-time fairness tracking
- ✅ Drift detection active
- ✅ Alert system configured
- ✅ Monitoring dashboard accessible
- ✅ Regular reports generated

---

## Integrated Workflow (Quick Path)

For users who want to run the complete workflow in one command:

### Quick Start

```bash
fairpipe run-pipeline \
  --config config.yml \
  --csv your_data.csv \
  --output-dir artifacts/ \
  --mlflow-experiment fairness_workflow
```

### Configuration File

Create `config.yml`:

```yaml
# Required: Sensitive attributes
sensitive: ["gender", "race"]

# Pipeline: Bias mitigation
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
  - name: repair
    transformer: "DisparateImpactRemover"
    params:
      features: ["score", "age"]
      sensitive: "gender"
      repair_level: 0.8

# Training: Fairness-aware model
training:
  method: "reductions"
  target_column: "y"
  params:
    constraint: "demographic_parity"
    eps: 0.01

# Validation: Threshold check
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
```

### What It Does

1. **Baseline Measurement**: Audits raw data for fairness issues
2. **Transform + Train**: Applies bias mitigation and trains fairness-aware model
3. **Final Validation**: Compares to baseline and validates against threshold

### Output

- Model artifacts
- Metrics and reports
- MLflow tracking (if configured)
- Validation results

See `demo_integrated.ipynb` for a complete example.

---

## Best Practices & Troubleshooting

### Best Practices

1. **Start Early**: Integrate fairness from the beginning, not as an afterthought
2. **Set Clear Thresholds**: Define fairness requirements upfront with stakeholders
3. **Use Statistical Validation**: Always include confidence intervals and effect sizes
4. **Document Everything**: Keep records of decisions, thresholds, and results
5. **Monitor Continuously**: Fairness can degrade over time in production
6. **Consider Intersectionality**: Analyze combinations of sensitive attributes
7. **Balance Trade-offs**: Understand fairness-accuracy trade-offs for your use case

### Common Issues

#### Issue: "Insufficient group size"

**Problem**: Groups have fewer than 30 samples.

**Solutions**:
- Increase `min_group_size` parameter (with caution)
- Collect more data for underrepresented groups
- Use stratified sampling
- Consider group merging (if legally/ethically appropriate)

#### Issue: "Validation failed - metric exceeds threshold"

**Problem**: Model doesn't meet fairness threshold.

**Solutions**:
- Adjust fairness constraint strength (e.g., increase `eps` for reductions)
- Try different mitigation transformers
- Explore fairness-accuracy trade-offs
- Revisit threshold (if appropriate with stakeholders)

#### Issue: "Proxy detection finds many features"

**Problem**: Many features are correlated with sensitive attributes.

**Solutions**:
- Review feature engineering process
- Consider removing high-correlation features
- Use `ProxyDropper` transformer
- Document proxy relationships

#### Issue: "Model accuracy decreased significantly"

**Problem**: Fairness constraints reduced model performance.

**Solutions**:
- Use Pareto Frontier to find optimal trade-off
- Try different training methods
- Adjust constraint tolerance
- Consider post-processing calibration

### Getting Help

- Check `demo_integrated.ipynb` for complete examples
- Review test files in `tests/` for usage patterns
- See `README.md` for module-specific documentation
- Check `CHANGELOG.md` for recent updates

---

## Advanced Topics

### Custom Metrics

Implement custom fairness metrics:

```python
from fairness_pipeline_dev_toolkit.metrics.base import BaseMetric

class CustomFairnessMetric(BaseMetric):
    def compute(self, y_true, y_pred, sensitive_features):
        # Your custom metric logic
        return result
```

### Custom Transformers

Create custom bias mitigation transformers:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, sensitive_features=None):
        # Fit logic
        return self
    
    def transform(self, X):
        # Transform logic
        return X_transformed
```

### Multi-Objective Optimization

Combine multiple fairness objectives:

```python
# Use Lagrangian trainer with multiple constraints
trainer = LagrangianFairnessTrainer(
    model=model,
    fairness="demographic_parity",  # Primary constraint
    # Can extend to support multiple constraints
    ...
)
```

### Integration with MLflow

Advanced MLflow logging:

```python
import mlflow
from fairness_pipeline_dev_toolkit.integration.mlflow_logger import log_workflow_results

with mlflow.start_run():
    # Log custom parameters
    mlflow.log_params({
        "fairness_threshold": 0.05,
        "training_method": "reductions",
        "constraint": "demographic_parity"
    })
    
    # Log workflow results
    log_workflow_results(
        baseline_metrics=baseline_results,
        final_metrics=final_results,
        validation_result=validation_result,
        model=model,
        config_path="config.yml"
    )
```

---

## Conclusion

This guide has walked you through using the Fairness Pipeline Development Toolkit across the complete model development cycle. The toolkit provides:

- **Comprehensive Coverage**: From data exploration to production monitoring
- **Statistical Rigor**: Confidence intervals, effect sizes, and hypothesis testing
- **Flexibility**: Use individual modules or integrated workflows
- **Production Ready**: Monitoring, alerting, and CI/CD integration

### Next Steps

1. **Try the Integrated Workflow**: Start with `fairpipe run-pipeline` for a quick end-to-end example
2. **Explore Demos**: Check out `demo_integrated.ipynb`, `demo_training.ipynb`, and `demo_monitoring.ipynb`
3. **Customize for Your Use Case**: Adapt configurations and workflows to your specific needs
4. **Set Up Monitoring**: Implement production monitoring early in your deployment

### Resources

- **README.md**: Module-specific documentation
- **Demo Notebooks**: Complete working examples
- **Test Suite**: Usage patterns and edge cases
- **ADR-001**: Architecture decision records

---

**Version**: 0.6.0  
**Last Updated**: 2026-01-26

For questions, issues, or contributions, please see [CONTRIBUTING.md](CONTRIBUTING.md).

