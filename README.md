# Fairness Pipeline Development Toolkit

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

### CLI Example

```
python -m fairness_pipeline_dev_toolkit.cli.main train \
  --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
  --csv dev_sample.csv \
  --profile training \
  --out-csv artifacts/training_output.csv
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

## CLI Usage
### 1️⃣ Fairness Validation

```console
python -m fairness_pipeline_dev_toolkit.cli.main validate \
  --csv dev_sample.csv --y-true y_true --y-pred y_pred \
  --sensitive sensitive --backend native \
  --with-ci --ci-level 0.95 --with-effects
```

### 2️⃣ Fair Pipeline Execution
```console
python -m fairness_pipeline_dev_toolkit.cli.main pipeline \
  --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
  --csv dev_sample.csv --out-csv artifacts/sample.transformed.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_run.md
```

### 3️⃣ Fair Model Training
```console
python -m fairness_pipeline_dev_toolkit.cli.main train \
  --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
  --csv dev_sample.csv --profile training
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
├── tests/
│   ├── training/
│   ├── pipeline/
│   └── system/
└── artifacts/
```

## Next Phase: Monitoring Module

The next release (v0.5.0) will introduce Monitoring — continuous drift detection, fairness drift tracking, and automated alerting over time windows.