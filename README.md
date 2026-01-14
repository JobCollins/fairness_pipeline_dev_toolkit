# Fairness Pipeline Development Toolkit

**Version:** 0.5.0

A unified, statistically-rigorous framework for **detecting**, **mitigating**, **training**, and **validating** fairness in ML workflows. The toolkit provides both **modular components** and an **integrated end-to-end workflow** spanning data-to-model fairness — enabling teams to move from ad-hoc checks to automated, continuous fairness assurance in CI/CD.

## What This Toolkit Is

The Fairness Pipeline Development Toolkit is a Python package designed for ML engineers, data scientists, and fairness practitioners who need to:

- **Measure fairness** in datasets and model predictions using statistical methods
- **Detect bias** in data through automated analysis (representation, statistical disparities, proxy variables)
- **Mitigate bias** using sklearn-compatible transformers (reweighing, disparate impact removal, proxy dropping)
- **Train fairness-aware models** using constraint-based methods (Fairlearn reductions, PyTorch regularizers, Lagrangian optimization)
- **Validate models** against fairness thresholds with statistical confidence intervals
- **Monitor fairness** in production with real-time tracking, drift detection, and alerting

## Who This Toolkit Is For

- **ML Engineers**: Building and deploying fair ML models
- **Data Engineers**: Implementing fairness checks in data pipelines
- **Fairness Practitioners**: Conducting fairness audits and assessments
- **Researchers**: Experimenting with fairness mitigation techniques
- **DevOps/CI/CD Teams**: Integrating fairness validation into automated pipelines

## What Problem It Solves

Traditional ML workflows often lack systematic fairness assessment, leading to:
- Models that perpetuate or amplify existing biases
- Unfair outcomes for protected groups
- Compliance risks with fairness regulations
- Lack of visibility into fairness metrics over time

This toolkit addresses these issues by providing:
- **Automated bias detection** before model training
- **Fairness-aware training methods** that embed constraints directly into learning
- **Statistical validation** with confidence intervals and effect sizes
- **Production monitoring** to detect fairness drift over time
- **CI/CD integration** for continuous fairness assurance

## Recommended Deployment and Usage Model

The toolkit is designed as a **Python package with CLI entry points**, supporting multiple usage patterns:

1. **CLI Usage (Primary)**: Run fairness checks and workflows from the command line
   ```bash
   fairpipe validate --csv data.csv --y-true y_true --y-pred y_pred --sensitive gender
   fairpipe run-pipeline --config config.yml --csv data.csv --output-dir artifacts/
   ```

2. **Programmatic Usage**: Import components into Python code
   ```python
   from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
   analyzer = FairnessAnalyzer()
   results = analyzer.demographic_parity_difference(y_pred, sensitive)
   ```

3. **Integrated Workflow**: End-to-end automation from raw data to validated model
   ```bash
   fairpipe run-pipeline --config config.yml --csv data.csv --output-dir artifacts/
   ```

The toolkit is **not** designed as:
- A web service or REST API (no HTTP endpoints)
- A long-running daemon (CLI runs once and exits)
- A distributed computing framework (single-threaded execution)
- A database-backed system (file-based I/O only)

---

## Getting Started (Local Setup)

This section provides step-by-step instructions to run the toolkit locally from scratch.

### Prerequisites

**System Requirements:**
- **Python**: 3.10 or higher (tested on Python 3.10, 3.11, 3.12)
- **Operating System**: macOS, Linux, or Windows
- **Package Manager**: `pip` (Python's package installer)
- **Optional**: `conda` (alternative package manager)

**System Dependencies:**
- No system-level dependencies required (all dependencies are Python packages)
- For PyTorch training: CUDA support optional (CPU-only works)

**Disk Space:**
- Minimum: ~500 MB for core installation
- With training extras: ~2 GB (includes PyTorch)
- With monitoring extras: ~1 GB (includes Streamlit, Dash, Plotly)

### Step 1: Clone or Download the Repository

If you have the repository:
```bash
cd fairness_pipeline_dev_toolkit
```

If you're installing from a package (future PyPI release):
```bash
# Not yet available on PyPI - install from source for now
```

### Step 2: Create a Virtual Environment

**Using venv (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n fairness_toolkit python=3.10
conda activate fairness_toolkit
```

### Step 3: Install Dependencies

**Option A: Core Installation (Minimum)**
```bash
# Install core toolkit with adapters (Fairlearn, Aequitas)
pip install -e .[adapters]
```

**Option B: Full Installation (Recommended)**
```bash
# Install core + training + monitoring extras
pip install -e .[adapters,training,monitoring]
```

**Option C: With Pinned Dependencies (Reproducible)**
```bash
# Install pip-tools first
pip install pip-tools

# Generate pinned requirements
pip-compile --extra training --extra monitoring --extra adapters \
    --output-file=requirements.txt requirements-dev.in

# Install from pinned requirements
pip install -r requirements.txt
```

**Note on PyTorch**: If you're installing the `training` extra, PyTorch will be installed automatically. For GPU support, you may need to install PyTorch separately first following instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Step 4: Verify Installation

Check that the CLI is available:
```bash
fairpipe version
```

You should see: `0.5.0`

If you get a `ModuleNotFoundError`, try:
```bash
python -m fairness_pipeline_dev_toolkit.cli.main version
```

### Step 5: Run Your First Example

**Example 1: Quick Fairness Validation**

Using the provided sample data:
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

This will:
- Load `dev_sample.csv`
- Compute demographic parity and equalized odds differences
- Calculate bootstrap confidence intervals
- Compute effect sizes
- Save a markdown report to `artifacts/validation_report.md`

**Example 2: Pipeline-Only Workflow**

Run bias detection and mitigation (without training):
```bash
fairpipe pipeline \
  --config pipeline.config.yml \
  --csv dev_sample.csv \
  --out-csv artifacts/transformed_data.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_report.md
```

This will:
- Detect bias in the data (representation, statistical, proxy analysis)
- Apply configured transformers (reweighing, disparate impact removal)
- Save transformed data to `artifacts/transformed_data.csv`
- Save detection results to `artifacts/detectors.json`
- Generate a report to `artifacts/pipeline_report.md`

**Example 3: Integrated Workflow (Full Pipeline)**

Run the complete three-step workflow:
```bash
# First, create a config.yml with training section
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

# Then run the integrated workflow
fairpipe run-pipeline \
  --config config.yml \
  --csv dev_sample.csv \
  --output-dir artifacts/workflow \
  --min-group-size 30
```

This will:
1. **Baseline Measurement**: Measure fairness on raw data
2. **Transform + Train**: Apply bias mitigation, then train a fairness-aware model
3. **Final Validation**: Compare final metrics to baseline and check threshold

### Step 6: Check Outputs

All outputs are saved to the `artifacts/` directory (or the directory you specify with `--output-dir`):

```
artifacts/
├── validation_report.md          # Fairness validation report
├── transformed_data.csv          # Data after bias mitigation
├── detectors.json                # Bias detection results
├── pipeline_report.md            # Pipeline execution report
└── workflow/                     # Integrated workflow outputs
    ├── workflow_results.json     # Complete workflow results
    ├── baseline_metrics.json     # Baseline fairness metrics
    ├── final_metrics.json         # Final model fairness metrics
    ├── model.pkl                 # Trained model (if applicable)
    └── config.yml                # Configuration used
```

### Step 7: Explore Demo Notebooks

The repository includes several demo notebooks:

- `demo_integrated.ipynb`: Complete integrated workflow example
- `demo_training.ipynb`: Training module examples
- `demo_monitoring.ipynb`: Production monitoring examples
- `demo_pipe.ipynb`: Pipeline module examples

To run notebooks:
```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook
```

Then open any of the `demo_*.ipynb` files.

---

## Usage Examples

### Minimal Example: Fairness Validation

```python
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

# Load data
df = pd.read_csv("data.csv")

# Initialize analyzer
analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

# Compute demographic parity difference
result = analyzer.demographic_parity_difference(
    y_pred=df["y_pred"].to_numpy(),
    sensitive=df["gender"].to_numpy(),
    with_ci=True,
    ci_level=0.95
)

print(f"Demographic Parity Difference: {result.value:.4f}")
print(f"95% CI: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
```

### Pipeline Example: Bias Detection and Mitigation

```python
import pandas as pd
from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    build_pipeline,
    apply_pipeline,
    run_detectors
)

# Load config and data
config = load_config("pipeline.config.yml")
df = pd.read_csv("data.csv")

# Run bias detection
detector_report = run_detectors(df=df, cfg=config)
print("Bias Detection Results:", detector_report.body)

# Build and apply pipeline
pipeline = build_pipeline(config)
transformed_df, _ = apply_pipeline(pipeline, df)

# Save results
transformed_df.to_csv("transformed_data.csv", index=False)
```

### Integrated Workflow Example

```python
import pandas as pd
from fairness_pipeline_dev_toolkit.integration.orchestrator import execute_workflow
from fairness_pipeline_dev_toolkit.pipeline.config import load_config

# Load config and data
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
else:
    print("❌ Validation FAILED")
    print(f"Reason: {result.validation_result.message}")
```

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
- **90+ tests** across all modules
- Integration tests for orchestrator and MLflow
- System tests for CLI end-to-end workflows
- Unit tests for individual components

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

- **Comprehensive Guide**: See [DOCS.md](DOCS.md) for detailed usage across the ML lifecycle
- **Architecture Decisions**: See [docs/ADR-001-architecture.md](docs/ADR-001-architecture.md)
- **Demo Notebooks**: Explore `demo_*.ipynb` for complete examples
- **Test Suite**: Review `tests/` for usage patterns and edge cases

---

**Version**: 0.5.0  
**Last Updated**: 2024
