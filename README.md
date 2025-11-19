# Fairness Pipeline Development Toolkit

**Version:** 0.4.2

A unified, statistically-rigorous framework for **detecting**, **mitigating**, **training**, and **validating** fairness in ML workflows.  
The toolkit provides modular components spanning data-to-model fairness — enabling teams to move from ad-hoc checks to automated, continuous fairness assurance in CI/CD.

---

## 🧩 Modules Overview

### **1. Measurement Module**
Implements fairness **metrics**, **statistical validation**, and **MLflow/pytest integration**.

**Features**
- Unified `FairnessAnalyzer` API with adapters for Fairlearn and Aequitas.  
- Metrics: demographic parity, equalized odds, MAE parity.  
- Intersectional analysis with `min_group_size`.  
- Statistical validation via bootstrap CIs and effect sizes.  
- CLI: `validate` for fairness audits.  

---

### **2. Pipeline Module**
Automates **bias detection**, **feature mitigation**, and **CI/CD fairness checks** for data engineering teams.

**Features**
- Bias Detection Engine (representation, statistical, and proxy analysis).  
- sklearn-compatible transformers:
  - `InstanceReweighting`
  - `DisparateImpactRemover`
  - `ReweighingTransformer`
  - `ProxyDropper`
- YAML-based orchestration with multiple profiles (`pipeline`, `training`).  
- CLI: `pipeline` for end-to-end mitigation and artifact generation.  

#### Minimal Pipeline YAML
```yaml
sensitive: ["sensitive"]
alpha: 0.05
pipeline:
  - name: reweigh
    transformer: InstanceReweighting
  - name: repair
    transformer: DisparateImpactRemover
    params:
      features: ["score"]
```

#### Config Schema Highlights
- `sensitive` (required): list of column names used for fairness analysis.
- `benchmarks` (optional): mapping of attribute → group → expected proportion.
- `pipeline`: ordered steps; each step needs a `transformer` key and optional `params` dict.
- Profiles are shallow-merged over top-level defaults; validation errors surface with helpful messages when keys are missing or mis-typed.

---

### **3. Training Module**
Enables **fair model training** by embedding fairness objectives directly into learning algorithms.

**Features**
- **ReductionsWrapper (scikit-learn):** wraps any estimator with `fairlearn.reductions.ExponentiatedGradient` for constraint-based training (e.g., Demographic Parity).  
- **FairnessRegularizer (PyTorch):** integrates fairness penalties (e.g., statistical dependence) into differentiable loss functions.  
- **LagrangianFairnessTrainer (PyTorch):** enforces fairness constraints via dual optimization (Lagrange multipliers).  
- **GroupFairnessCalibrator:** applies Platt Scaling or Isotonic Regression post-training to balance probabilities across groups.  
- **ParetoFrontier Visualization Tool:** visualizes the fairness–accuracy trade-off to guide stakeholder decisions.

**Usage Example (PyTorch Regularizer)**
```python
from fairness_pipeline_dev_toolkit.training.torch_.losses import FairnessRegularizerLoss
from fairness_pipeline_dev_toolkit.training.torch_.lagrangian import LagrangianFairnessTrainer
```

---

### **4. Monitoring Module**
Enables **continuous fairness monitoring**, **drift detection**, and **automated alerting** for production ML systems.

**Features**
- **RealTimeFairnessTracker:** sliding-window metric computation with configurable window sizes.  
- **FairnessDriftAndAlertEngine:** KS-test based drift detection with optional wavelet decomposition for multi-scale analysis.  
- **FairnessReportingDashboard:** Plotly-based visualizations and Markdown report generation.  
- **FairnessABTestAnalyzer:** A/B testing utilities for fairness comparisons.  
- **Streamlit/Dash Apps:** interactive dashboards for real-time monitoring (see `apps/monitoring_streamlit_app.py` and `apps/monitoring_dash_app.py`).  

**Usage Example**
```python
from fairness_pipeline_dev_toolkit.monitoring import (
    RealTimeFairnessTracker,
    FairnessDriftAndAlertEngine,
    ColumnMap,
    TrackerConfig,
)

tracker = RealTimeFairnessTracker(
    TrackerConfig(window_size=10_000, min_group_size=30),
    artifacts_dir="artifacts/monitoring"
)
cmap = ColumnMap(
    y_pred="predictions",
    y_true="labels",
    protected=["gender", "race"],
    intersections=[["gender", "race"]]
)
tracker.process_batch(df, cmap)

drift_engine = FairnessDriftAndAlertEngine(DriftConfig())
alerts = drift_engine.analyze(tracker.metrics_ts)
```

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[adapters]
# Optional extras
pip install -e .[training,monitoring]
```

To install developer tooling without pins:
```bash
pip install -r requirements-dev.in
```

To generate pinned requirements with `pip-tools`:
```bash
pip install pip-tools
pip-compile --extra training --extra monitoring --extra adapters \
    --output-file=requirements.txt requirements-dev.in
pip install -r requirements.txt
```

> ⚠️ PyTorch wheels depend on platform/accelerator support. Follow the commands from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) before enabling the `training` extra.

pre-commit hooks ensure consistent formatting:
```bash
pre-commit install
```

### Pre-commit Hooks
- The repository includes `.pre-commit-config.yaml` with `ruff`, `black`, `isort`, and `nbstripout`.
- Run `pre-commit install` once per clone (see installation above) to enable auto-formatting and notebook sanitisation on every commit.

## Quick Start

After installation, run a quick fairness validation:

```bash
# Check version
fairpipe version

# Validate fairness metrics on a dataset
fairpipe validate \
  --csv data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --with-ci --with-effects

# Run bias detection and mitigation pipeline
fairpipe pipeline \
  --config pipeline.config.yml \
  --csv data.csv \
  --out-csv output.csv
```

## CLI Usage

> **Note:** The `fairpipe` command is available as a shorthand entry point. You can also use the full form: `python -m fairness_pipeline_dev_toolkit.cli.main <command>`. The entry point is defined in `pyproject.toml`.

### 1️⃣ Fairness Validation

```console
fairpipe validate \
  --csv dev_sample.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive sensitive \
  --backend native \
  --with-ci \
  --ci-level 0.95 \
  --with-effects \
  --out report.md
```

### 2️⃣ Fair Pipeline Execution

```console
fairpipe pipeline \
  --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
  --csv dev_sample.csv \
  --out-csv artifacts/sample.transformed.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_run.md
```

### 3️⃣ Fair Model Training

**Train with Regularizer (Pareto Frontier):**
```console
fairpipe train-regularized \
  --csv data.csv \
  --etas "0.0,0.2,0.5,1.0" \
  --epochs 50 \
  --lr 1e-3 \
  --out-json artifacts/pareto_points.json \
  --out-png artifacts/pareto.png
```

**Train with Lagrangian Constraints:**
```console
fairpipe train-lagrangian \
  --csv data.csv \
  --fairness demographic_parity \
  --dp-tol 0.02 \
  --epochs 100 \
  --batch-size 128 \
  --out-json artifacts/training_history.json
```

### 4️⃣ Post-Training Calibration

```console
fairpipe calibrate \
  --csv scores.csv \
  --method platt \
  --min-samples 20 \
  --out-csv artifacts/calibrated_scores.csv
```

### 5️⃣ Utility Commands

**Check version:**
```console
fairpipe version
```

**Pre-commit sample check:**
```console
fairpipe sample-check
```

## Testing & Validation

Run all tests:

```console
pytest -q
```


System test for pipeline:

```console
pytest tests/system/test_cli_e2e_pipeline.py::test_cli_pipeline_e2e[native] -q
```


Training module tests:

```console
pytest tests/training -q

```

## Repository Structure

```
fairness_pipeline_dev_toolkit/
├── cli/
├── measurement/
├── metrics/
├── stats/
├── pipeline/
│   ├── config/
│   ├── detectors/
│   ├── orchestration/
│   ├── transformers/
│   └── pipeline.config.yml
├── training/
│   ├── sklearn_/              # ReductionsWrapper
│   ├── torch_/                # Loss + LagrangianTrainer
│   ├── postproc/              # GroupFairnessCalibrator
│   ├── viz/                   # Pareto Frontier Visualization
│   └── __init__.py
├── monitoring/
│   ├── tracker.py             # RealTimeFairnessTracker
│   ├── drift.py               # FairnessDriftAndAlertEngine
│   ├── dashboard.py           # FairnessReportingDashboard
│   ├── abtest.py              # FairnessABTestAnalyzer
│   └── config.py              # MonitoringSettings, DriftConfig
├── apps/
│   ├── monitoring_streamlit_app.py
│   └── monitoring_dash_app.py
├── tests/
│   ├── training/
│   ├── pipeline/
│   ├── monitoring/
│   └── system/
└── artifacts/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and formatting (enforced via pre-commit hooks)
- Testing requirements
- Pull request process

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.