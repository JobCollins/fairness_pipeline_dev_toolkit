# Fairness Pipeline Development Toolkit

**Version:** 0.6.0  
**Status:** Production-ready | Available on [PyPI](https://pypi.org/project/fairpipe/)  
![Coverage](https://img.shields.io/badge/coverage-86%25-green)


A unified, statistically-rigorous framework for **detecting**, **mitigating**, **training**, and **validating** fairness in ML workflows. The toolkit provides both **modular components** and an **integrated end-to-end workflow** spanning data-to-model fairness — enabling teams to move from ad-hoc checks to automated, continuous fairness assurance in CI/CD.

---

## Quick Install

### From PyPI (Recommended for Production)

The toolkit is available on PyPI and can be installed with pip:

```bash
pip install fairpipe
```

This installs the core package with all essential dependencies. For optional features, see [Installation Options](#installation-options) below.

## Installation Options

### Core Installation (Default)

The base installation includes all essential fairness measurement and pipeline components:

```bash
pip install fairpipe
```

**Included:**
- Fairness metrics computation (demographic parity, equalized odds, etc.)
- Bias detection and mitigation transformers
- Statistical validation (bootstrap CIs, effect sizes)
- Pipeline orchestration
- Integration with scikit-learn

### Optional Extras

Install additional features using extra dependency groups:

```bash
# Training methods (PyTorch-based fairness-aware training)
pip install fairpipe[training]

# Production monitoring tools (dashboards and drift detection)
pip install fairpipe[monitoring]

# External metric backends (Fairlearn, Aequitas adapters)
pip install fairpipe[adapters]

# Install all optional dependencies
pip install fairpipe[training,monitoring,adapters]
```

**Optional dependency groups:**
- **`training`**: PyTorch-based training methods (regularized loss, Lagrangian constraints, calibration)
- **`monitoring`**: Production monitoring tools (Streamlit/Dash dashboards, drift detection, alerting)
- **`adapters`**: External metric backends (Fairlearn, Aequitas) for compatibility with existing tools

### Development Installation

For development or to use the latest features from source:

```bash
git clone https://github.com/SvrusIO/fAIr
cd fAIr
pip install -e ".[training,monitoring,adapters,dev]"
```

### System Requirements

- **Python**: 3.10 or higher (tested on 3.10, 3.11, 3.12)
- **Operating System**: macOS, Linux, or Windows
- **Disk Space**: 
  - Core: ~500 MB
  - With training: ~2 GB
  - With monitoring: ~1 GB

**Note on PyTorch**: If installing the `training` extra, PyTorch will be installed automatically. For GPU support, install PyTorch separately following instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## Quick Start

### 1. Install the Package

```bash
pip install fairpipe
```

### 2. Quick CLI Usage

Run a quick fairness validation on your predictions:

```bash
fairpipe validate \
  --csv data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --with-ci \
  --out report.md
```

Run the complete integrated workflow (baseline → transform+train → validate):

```bash
fairpipe run-pipeline \
  --config config.yml \
  --csv data.csv \
  --output-dir artifacts/
```

### 3. Quick Python Usage

```python
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
import pandas as pd

# Load your data
df = pd.read_csv("data.csv")

# Initialize analyzer
analyzer = FairnessAnalyzer(min_group_size=30)

# Compute demographic parity difference with confidence intervals
result = analyzer.demographic_parity_difference(
    y_pred=df["y_pred"].to_numpy(),
    sensitive=df["gender"].to_numpy(),
    with_ci=True
)

print(f"DPD: {result.value:.4f}")
print(f"95% CI: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
```

For more examples, see [Usage Examples](#usage-examples) below or check the [Integration Guide](docs/integration_guide.md).

---

## Usage Examples

### Example 1: Fairness Validation

Validate fairness metrics on predictions with confidence intervals and effect sizes.

**CLI:**
```bash
fairpipe validate \
  --csv dev_sample.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive sensitive \
  --with-ci \
  --with-effects \
  --out artifacts/validation_report.md
```

**Python:**
```python
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

# Load data
df = pd.read_csv("data.csv")

# Initialize analyzer
analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

# Compute demographic parity difference with confidence intervals
result = analyzer.demographic_parity_difference(
    y_pred=df["y_pred"].to_numpy(),
    sensitive=df["gender"].to_numpy(),
    with_ci=True,
    ci_level=0.95
)

print(f"Demographic Parity Difference: {result.value:.4f}")
print(f"95% CI: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
print(f"Group sizes: {result.n_per_group}")
```

### Example 2: Bias Detection and Mitigation Pipeline

Detect bias in data and apply mitigation transformers.

**CLI:**
```bash
fairpipe pipeline \
  --config pipeline.config.yml \
  --csv dev_sample.csv \
  --out-csv artifacts/transformed_data.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_report.md
```

**Python:**
```python
import pandas as pd
from fairness_pipeline_dev_toolkit.pipeline import (
    load_config,
    build_pipeline,
    apply_pipeline,
    run_detectors
)

# Load configuration
config = load_config("pipeline.config.yml")
df = pd.read_csv("data.csv")

# Step 1: Run bias detection
detector_report = run_detectors(df=df, cfg=config)
print("Bias Detection Results:", detector_report.body)

# Step 2: Build and apply mitigation pipeline
pipeline = build_pipeline(config)
transformed_df, _ = apply_pipeline(pipeline, df)
transformed_df.to_csv("transformed_data.csv", index=False)
```

### Example 3: Integrated Workflow (Baseline → Transform+Train → Validate)

Run the complete end-to-end workflow: baseline measurement, data transformation, model training, and validation.

**CLI:**
```bash
# Create config.yml
cat > config.yml << EOF
sensitive: ["sensitive"]
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
training:
  method: "reductions"
  target_column: "y_true"
  params:
    constraint: "demographic_parity"
    eps: 0.01
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
EOF

# Run workflow
fairpipe run-pipeline \
  --config config.yml \
  --csv dev_sample.csv \
  --output-dir artifacts/workflow \
  --min-group-size 30
```

**Python:**
```python
import pandas as pd
from fairness_pipeline_dev_toolkit.integration import execute_workflow
from fairness_pipeline_dev_toolkit.pipeline import load_config

# Load configuration and data
config = load_config("config.yml")
df = pd.read_csv("data.csv")

# Execute complete workflow
result = execute_workflow(
    config=config,
    df=df,
    output_dir="artifacts/workflow",
    min_group_size=30,
    train_size=0.8
)

# Check validation result
if result.validation_result.passed:
    print("✅ Validation PASSED")
    print(f"Improvement: {result.validation_result.improvement:.4f}")
else:
    print("❌ Validation FAILED")
    print(f"Reason: {result.validation_result.message}")
    print(f"Baseline: {result.validation_result.baseline_metric_value:.4f}")
    print(f"Final: {result.validation_result.final_metric_value:.4f}")
```

---

## Public API

### Core Components

**Metrics:**
- `fairness_pipeline_dev_toolkit.metrics.FairnessAnalyzer` - Main class for computing fairness metrics
- `fairness_pipeline_dev_toolkit.metrics.MetricResult` - Result object containing metric values and metadata

**Pipeline:**
- `fairness_pipeline_dev_toolkit.pipeline.config.PipelineConfig` - Configuration dataclass
- `fairness_pipeline_dev_toolkit.pipeline.config.load_config` - Load configuration from YAML
- `fairness_pipeline_dev_toolkit.pipeline.build_pipeline` - Build pipeline from config
- `fairness_pipeline_dev_toolkit.pipeline.apply_pipeline` - Apply pipeline to data
- `fairness_pipeline_dev_toolkit.pipeline.run_detectors` - Run bias detection

**Transformers:**
- `fairness_pipeline_dev_toolkit.pipeline.InstanceReweighting` - Instance reweighing transformer
- `fairness_pipeline_dev_toolkit.pipeline.DisparateImpactRemover` - Disparate impact removal
- `fairness_pipeline_dev_toolkit.pipeline.ReweighingTransformer` - Reweighing transformer
- `fairness_pipeline_dev_toolkit.pipeline.ProxyDropper` - Proxy variable dropper

**Integration:**
- `fairness_pipeline_dev_toolkit.integration.execute_workflow` - Execute end-to-end workflow
- `fairness_pipeline_dev_toolkit.integration.WorkflowResult` - Workflow execution result
- `fairness_pipeline_dev_toolkit.integration.ValidationResult` - Validation result

**Training:**
- `fairness_pipeline_dev_toolkit.training.ReductionsWrapper` - Fairlearn reductions wrapper
- `fairness_pipeline_dev_toolkit.training.FairnessRegularizerLoss` - PyTorch fairness regularizer
- `fairness_pipeline_dev_toolkit.training.LagrangianFairnessTrainer` - Lagrangian constraint trainer
- `fairness_pipeline_dev_toolkit.training.GroupFairnessCalibrator` - Group-specific calibration

**Monitoring:**
- `fairness_pipeline_dev_toolkit.monitoring.RealTimeFairnessTracker` - Real-time metric tracking
- `fairness_pipeline_dev_toolkit.monitoring.FairnessDriftAndAlertEngine` - Drift detection and alerting
- `fairness_pipeline_dev_toolkit.monitoring.FairnessReportingDashboard` - Reporting dashboard

**Exceptions:**
- `fairness_pipeline_dev_toolkit.exceptions.FairnessToolkitError` - Base exception
- `fairness_pipeline_dev_toolkit.exceptions.ConfigValidationError` - Configuration validation error
- `fairness_pipeline_dev_toolkit.exceptions.MetricComputationError` - Metric computation error
- `fairness_pipeline_dev_toolkit.exceptions.PipelineExecutionError` - Pipeline execution error

See [API Reference](docs/api.md) for complete documentation.

---

## CLI Commands Reference

### `fairpipe version`
Print the toolkit version.

### `fairpipe validate`
Run fairness validation on a CSV file.

```bash
fairpipe validate \
  --csv data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --min-group-size 30 \
  --with-ci \
  --ci-level 0.95 \
  --with-effects \
  --out report.md
```

**Required arguments:**
- `--csv`: Path to CSV file
- `--y-true`: Column name for ground-truth labels
- `--sensitive`: Sensitive attribute column(s) (can specify multiple)

**Optional arguments:**
- `--y-pred`: Column name for predicted labels (classification)
- `--score`: Column name for predicted scores (regression)
- `--min-group-size`: Minimum samples per group (default: 30)
- `--backend`: Backend selection (`auto`, `native`, `fairlearn`, `aequitas`)
- `--with-ci`: Compute bootstrap confidence intervals
- `--ci-level`: Confidence level (default: 0.95)
- `--bootstrap-B`: Number of bootstrap samples (default: 1000)
- `--with-effects`: Compute effect sizes
- `--out`: Path to save markdown report

### `fairpipe pipeline`
Run bias detection and mitigation pipeline (without training).

```bash
fairpipe pipeline \
  --config pipeline.config.yml \
  --csv data.csv \
  --out-csv output.csv \
  --detector-json detectors.json \
  --report-md report.md \
  --no-detectors  # Skip bias detection
```

**Required arguments:**
- `--config`: Path to pipeline configuration YAML
- `--csv`: Path to input CSV file

**Optional arguments:**
- `--profile`: Config profile name (if YAML has profiles)
- `--out-csv`: Path to save transformed CSV
- `--detector-json`: Path to save detector results JSON
- `--report-md`: Path to save markdown report
- `--no-detectors`: Skip bias detection stage

### `fairpipe run-pipeline`
Execute integrated three-step workflow (baseline → transform+train → validate).

```bash
fairpipe run-pipeline \
  --config config.yml \
  --csv data.csv \
  --output-dir artifacts/ \
  --min-group-size 30 \
  --train-size 0.8 \
  --mlflow-experiment fairness_workflow \
  --mlflow-run-name run_001
```

**Required arguments:**
- `--config`: Path to config YAML (must include `training` section)
- `--csv`: Path to input CSV file

**Optional arguments:**
- `--profile`: Config profile name
- `--output-dir`: Directory to save artifacts
- `--min-group-size`: Minimum samples per group (default: 30)
- `--train-size`: Proportion of data for training (default: 0.8)
- `--mlflow-experiment`: MLflow experiment name (enables MLflow logging)
- `--mlflow-run-name`: MLflow run name

**Exit codes:**
- `0`: Validation passed (metrics meet threshold)
- `1`: Validation failed (metrics exceed threshold) or error occurred

### `fairpipe train-regularized`
Train a neural network with fairness regularizer and generate Pareto frontier.

```bash
fairpipe train-regularized \
  --csv data.csv \
  --etas "0.0,0.2,0.5,1.0" \
  --epochs 50 \
  --lr 1e-3 \
  --out-json pareto_points.json \
  --out-png pareto.png
```

**Required CSV columns:** `f0`, `f1`, ..., `y`, `s` (features, label, sensitive)

### `fairpipe train-lagrangian`
Train a neural network with Lagrangian fairness constraints.

```bash
fairpipe train-lagrangian \
  --csv data.csv \
  --fairness demographic_parity \
  --dp-tol 0.02 \
  --epochs 100 \
  --batch-size 128 \
  --out-json training_history.json
```

### `fairpipe calibrate`
Apply group-specific calibration to prediction scores.

```bash
fairpipe calibrate \
  --csv scores.csv \
  --method platt \
  --min-samples 20 \
  --out-csv calibrated_scores.csv
```

**Required CSV columns:** `score`, `y`, `g` (scores, labels, groups)

### `fairpipe sample-check`
Lightweight pre-commit check for sample data existence.

```bash
fairpipe sample-check
```

---

## Configuration Guide

### Pipeline Configuration (`pipeline.config.yml`)

Minimal configuration:
```yaml
sensitive: ["sensitive"]
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
  - name: repair
    transformer: "DisparateImpactRemover"
    params:
      features: ["score"]
      sensitive: "sensitive"
      repair_level: 0.8
```

Full configuration with profiles:
```yaml
sensitive: ["gender", "race"]
benchmarks:
  gender:
    M: 0.5
    F: 0.5
alpha: 0.05
proxy_threshold: 0.30

pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
  - name: repair
    transformer: "DisparateImpactRemover"
    params:
      features: ["score", "age"]
      sensitive: "gender"
      repair_level: 0.8

profiles:
  training:
    pipeline:
      - name: reweigh
        transformer: "InstanceReweighting"
```

### Integrated Workflow Configuration (`config.yml`)

Configuration for `fairpipe run-pipeline` must include a `training` section:

```yaml
sensitive: ["sensitive"]
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"

training:
  method: "reductions"  # Options: "reductions", "regularized", "lagrangian"
  target_column: "y"
  params:
    constraint: "demographic_parity"  # For reductions method
    eps: 0.01
    T: 50

fairness_metric: "demographic_parity_difference"
validation_threshold: 0.05
```

**Training method options:**

1. **`reductions`** (scikit-learn): Uses Fairlearn's ExponentiatedGradient
   ```yaml
   training:
     method: "reductions"
     target_column: "y"
     params:
       constraint: "demographic_parity"  # or "equalized_odds"
       eps: 0.01
       T: 50
       base_estimator: null  # Default: LogisticRegression
   ```

2. **`regularized`** (PyTorch): Fairness penalty in loss function
   ```yaml
   training:
     method: "regularized"
     target_column: "y"
     params:
       eta: 0.5
       epochs: 10
       lr: 0.001
       device: "cpu"  # or "cuda"
   ```

3. **`lagrangian`** (PyTorch): Dual optimization with constraints
   ```yaml
   training:
     method: "lagrangian"
     target_column: "y"
     params:
       fairness: "demographic_parity"  # or "equal_opportunity"
       dp_tol: 0.02
       eo_tol: 0.02
       model_lr: 0.001
       lambda_lr: 0.01
       epochs: 10
       batch_size: 128
       device: "cpu"
   ```

### Environment Variables

The toolkit supports configuration via environment variables:

```bash
export FAIRPIPE_CONFIG_PATH="config.yml"
export FAIRPIPE_MIN_GROUP_SIZE=30
export FAIRPIPE_MLFLOW_EXPERIMENT="fairness_workflow"
```

See [Integration Guide](docs/integration_guide.md) for more details.

---

## Modules Overview

### 1. Measurement Module

**Purpose**: Compute fairness metrics with statistical validation.

**Key Components:**
- `FairnessAnalyzer`: Unified API for fairness metrics
- Adapters: `native`, `fairlearn`, `aequitas`
- Metrics: demographic parity, equalized odds, MAE parity
- Statistical validation: bootstrap CIs, effect sizes

**Usage:**
```python
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

analyzer = FairnessAnalyzer(min_group_size=30, backend="native")
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True
)
```

### 2. Pipeline Module

**Purpose**: Detect and mitigate bias in data.

**Key Components:**
- **Detectors**: Representation, statistical, proxy analysis
- **Transformers**: `InstanceReweighting`, `DisparateImpactRemover`, `ProxyDropper`, `ReweighingTransformer`
- **Orchestration**: YAML-based pipeline configuration

**Usage:**
```bash
fairpipe pipeline --config pipeline.config.yml --csv data.csv --out-csv output.csv
```

### 3. Training Module

**Purpose**: Train fairness-aware models.

**Key Components:**
- `ReductionsWrapper`: Fairlearn integration for scikit-learn
- `FairnessRegularizerLoss`: PyTorch loss with fairness penalty
- `LagrangianFairnessTrainer`: Constraint-based PyTorch training
- `GroupFairnessCalibrator`: Post-training calibration
- Pareto frontier visualization

**Usage:**
```python
from fairness_pipeline_dev_toolkit.training import ReductionsWrapper
from sklearn.linear_model import LogisticRegression

model = ReductionsWrapper(
    LogisticRegression(),
    constraint="demographic_parity",
    eps=0.01
)
model.fit(X_train, y_train, sensitive_features=A_train)
```

### 4. Monitoring Module

**Purpose**: Monitor fairness in production.

**Key Components:**
- `RealTimeFairnessTracker`: Sliding-window metric computation
- `FairnessDriftAndAlertEngine`: KS-test based drift detection
- `FairnessReportingDashboard`: Plotly visualizations and reports
- `FairnessABTestAnalyzer`: A/B testing utilities
- Streamlit/Dash apps: Interactive dashboards

**Usage:**
```python
from fairness_pipeline_dev_toolkit.monitoring import RealTimeFairnessTracker, TrackerConfig

tracker = RealTimeFairnessTracker(
    TrackerConfig(window_size=10_000, min_group_size=30),
    artifacts_dir="artifacts/monitoring"
)
tracker.process_batch(df, column_map)
```

### 5. Integration Module

**Purpose**: Orchestrate end-to-end workflows.

**Key Components:**
- `execute_workflow`: Three-step workflow orchestrator
- `log_workflow_results`: MLflow integration
- `generate_validation_report`: Report generation

**Usage:**
```bash
fairpipe run-pipeline --config config.yml --csv data.csv --output-dir artifacts/
```

---

## Limitations and Non-Goals

### Known Limitations

1. **File-Based I/O Only**
   - Input/output assumes CSV files
   - No database connectors (SQL, Parquet, etc.)
   - No streaming data support

2. **Single-Threaded Execution**
   - All processing is single-threaded/single-process
   - No support for distributed computing (Spark, Dask, Ray)
   - Large datasets may require external orchestration

3. **No Service Layer**
   - CLI runs once and exits (no long-running service)
   - No REST API or HTTP endpoints
   - No job queue or scheduling

4. **Limited Error Handling**
   - Some functions raise generic exceptions
   - No structured error types for programmatic handling
   - Error messages may not always be user-friendly

5. **Platform-Specific Dependencies**
   - Aequitas adapter requires Python < 3.12
   - PyTorch installation varies by platform/accelerator
   - Some features may not work on all operating systems

6. **Statistical Limitations**
   - Bootstrap CIs can be unstable for very small samples
   - Effect sizes may be unreliable with insufficient group sizes
   - Minimum group size of 30 is recommended but not enforced

### Non-Goals

The toolkit is **not** designed to:
- Provide a web UI or dashboard (monitoring apps are separate)
- Support real-time streaming inference (batch processing only)
- Replace domain expertise in fairness assessment
- Guarantee legal compliance (consult legal experts)
- Handle all types of bias (focuses on group fairness)
- Support all ML frameworks (scikit-learn and PyTorch only)

### Experimental/Unstable Features

1. **Wavelet-based drift detection**: Optional feature in monitoring module, may be unstable
2. **Aequitas adapter**: Requires Python < 3.12, may have compatibility issues
3. **Proxy detection**: Correlation-based proxy detection may have false positives
4. **Intersectional analysis**: Requires careful group size management

---

## Testing

Run the test suite:
```bash
pytest -q
```

Run specific test suites:
```bash
pytest tests/integration/ -q
pytest tests/system/ -q
pytest tests/pipeline/ -q
pytest tests/training/ -q
pytest tests/monitoring/ -q
```

The test suite includes:
- **673 tests** across all modules with **86% code coverage**
- Integration tests for orchestrator and MLflow
- Expanded integration tests with comprehensive edge case coverage
- Property-based tests using Hypothesis for statistical invariants
- System tests for CLI end-to-end workflows
- Unit tests for individual components
- Comprehensive coverage of detectors, transformers, metrics, and training modules
- Statistical validation tests for bootstrap CIs, effect sizes, and multiple testing corrections

---

## Repository Structure

```
fairness_pipeline_dev_toolkit/
├── fairness_pipeline_dev_toolkit/    # Main package
│   ├── cli/                          # CLI commands
│   ├── integration/                  # Workflow orchestrator, MLflow, reporting
│   ├── measurement/                  # FairnessAnalyzer API
│   ├── metrics/                      # Core metrics + adapters
│   ├── pipeline/                     # Transformers, detectors, config
│   ├── training/                     # sklearn/PyTorch training methods
│   ├── monitoring/                   # Production monitoring tools
│   ├── stats/                        # Statistical validation
│   └── utils/                        # Shared utilities
├── tests/                            # Test suite
├── artifacts/                        # Generated outputs (gitignored)
├── apps/                             # Monitoring dashboards (Streamlit/Dash)
├── scripts/                          # Utility scripts
├── demo_*.ipynb                      # Demo notebooks
├── config.yml                        # Example integrated workflow config
├── pipeline.config.yml               # Example pipeline config
└── requirements.txt                  # Pinned dependencies
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and formatting (enforced via pre-commit hooks)
- Testing requirements
- Pull request process

### Pre-commit Hooks

The repository includes `.pre-commit-config.yaml` with `ruff`, `black`, `isort`, and `nbstripout`.

To enable:
```bash
pre-commit install
```

This ensures consistent formatting and notebook sanitization on every commit.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Additional Resources

- **API Reference**: See [docs/api.md](docs/api.md) for complete API documentation
- **Integration Guide**: See [docs/integration_guide.md](docs/integration_guide.md) for integration examples
- **Versioning Strategy**: See [docs/VERSIONING.md](docs/VERSIONING.md) for versioning and backward compatibility policy
- **Architecture Decisions**: See [docs/ADR-001-architecture.md](docs/ADR-001-architecture.md)
- **Comprehensive Guide**: See [DOCS.md](DOCS.md) for detailed usage across the ML lifecycle
- **Documentation Site**: Automated documentation builds available via GitHub Pages (see `.github/workflows/docs.yml`)
- **Security**: See [SECURITY.md](SECURITY.md) for security policy and [.github/SECURITY_REVIEW_PROCESS.md](.github/SECURITY_REVIEW_PROCESS.md) for security review process
- **Demo Notebooks**: Explore `demo_*.ipynb` for complete examples
- **Test Suite**: Review `tests/` for usage patterns and edge cases

---

**Version**: 0.6.0  
**Last Updated**: 2026-01-26
