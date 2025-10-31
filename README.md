# Fairness Pipeline Development Toolkit

A unified, statistically-rigorous framework for **detecting**, **mitigating**, and **validating** bias in ML workflows.  
The toolkit provides modular components for fairness **measurement** and **pipeline integration**, enabling teams to move from ad-hoc fairness checks to automated, continuous validation within CI/CD.

---

## 🧩 Modules Overview

### **1. Measurement Module**
Implements fairness **metrics**, **statistical validation**, and **MLflow/pytest integration**.

**Features**
- Unified `FairnessAnalyzer` API with adapters for Fairlearn and Aequitas.  
- Classification and regression metrics (e.g., demographic parity, equalized odds, MAE parity).  
- Intersectional group analysis with `min_group_size`.  
- Statistical validation: bootstrap CIs (95%), effect sizes (risk ratios).  
- Integrated workflow: MLflow logging + pytest assertions.  
- CLI command: `validate` for quick fairness audits.

---

### **2. Pipeline Module**
Automates **bias detection**, **feature mitigation**, and **CI/CD fairness checks** for data engineering teams.

**Features**
- Bias Detection Engine (representation, statistical, proxy analysis).  
- Modular, sklearn-compatible transformers:  
  - `InstanceReweighting`  
  - `DisparateImpactRemover`  
  - `ReweighingTransformer`  
  - `ProxyDropper`  
- Configurable YAML-based pipeline orchestration.  
- Automated fairness validation in CI/CD (pytest + GitHub Actions).  
- Generates `BiasReport` JSON + Markdown summary.

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

## 🚀 CLI Usage

### 1️⃣ Fairness Measurement (from Measurement Module)

Validate metrics on any CSV dataset:

```console
python -m fairness_pipeline_dev_toolkit.cli.main validate \
  --csv dev_sample.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive sensitive \
  --backend native \
  --min-group-size 2 \
  --with-ci --ci-level 0.95 --bootstrap-B 1000 \
  --with-effects \
  --out artifacts/report_with_ci.md
```


**Example Output**
```console

| Metric | Value | CI (95%) | Effect Size | n_per_group |
|---------|--------|----------|--------------|--------------|
| demographic_parity_difference | 0.300 | [0.0282, 0.0967] | 2.50 | {"A": 98, "B": 62, "C": 40} |
| equalized_odds_difference | 0.3956 | [0.1424, 0.1832] | 3.77 | {"A": 98, "B": 62, "C": 40} |

```

### 2️⃣ Fair Pipeline Execution (from Pipeline Module)

Run full detection → transformation → report generation:

```console
python -m fairness_pipeline_dev_toolkit.cli.main pipeline \
  --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
  --csv dev_sample.csv \
  --out-csv artifacts/sample.transformed.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_run.md
```


Produces:

Transformed dataset (`sample.transformed.csv`)

Detector JSON (`detectors.json`) with top-level `"meta"` and `"body"`

Markdown report summarizing representation bias, disparities, and proxies

🧱 Repository Structure
```console
fairness_pipeline_dev_toolkit/
├── cli/                        # CLI entrypoints (validate / pipeline)
├── measurement/                # Measurement module core
├── metrics/                    # Fairness metrics & adapters
├── stats/                      # Statistical validation (bootstrap, effect size)
├── pipeline/
│   ├── config/                 # YAML loader and schema
│   ├── detectors/              # Bias detection engines + BiasReport
│   ├── transformers/           # sklearn-compatible mitigation transforms
│   ├── orchestration/          # Build + execute pipelines
│   └── pipeline.config.yml     # Example configuration
├── tests/                      # Unit + system tests
│   ├── pipeline/
│   └── system/
└── artifacts/                  # Generated reports & transformed outputs
```

## 🧪 Testing & Validation

Run full suite:

```console
pytest -q
```


Run CI smoke (simulates GitHub Actions):

```console
pytest tests/system/test_cli_e2e_pipeline.py::test_cli_pipeline_e2e[native] -q
```

### Example Configuration

```markdown
sensitive: ["group"]
alpha: 0.05
proxy_threshold: 0.30
report_out: "artifacts/detectors.json"
benchmarks:
  group: {A: 0.5, B: 0.5}
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
  - name: di
    transformer: "DisparateImpactRemover"
    params:
      features: ["x1"]
      sensitive: "group"
      repair_level: 0.8
```


### BiasReport JSON Schema
```json
{
  "meta": {
    "phase": "0",
    "alpha": 0.05,
    "proxy_threshold": 0.3
  },
  "body": {
    "summary": {...},
    "representation": [...],
    "disparities": [...],
    "proxies": [...]
  }
}
```

### 📈 Changelog Highlights (v0.1.0 → v0.2.0)
Added

- Pipeline Module: bias detection + mitigation integration.

- sklearn transformers (InstanceReweighting, DisparateImpactRemover, ReweighingTransformer, ProxyDropper).

- BiasReport JSON structure with meta and body.

- CLI command: pipeline – end-to-end data fairness orchestration.

- CI/CD automation + system tests for full pipeline validation.

- Demo notebook (demo.ipynb) showing combined workflow.