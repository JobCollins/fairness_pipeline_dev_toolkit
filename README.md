# Fairness Pipeline Development Toolkit — Measurement Module

A unified, statistically-rigorous **fairness toolkit** for ML workflows, delivered in two parts:

- **Measurement Module** – compute fairness metrics (DP, EO, MAE parity), bootstrap CIs, and effect sizes; integrates with MLflow, pytest, and CI.
- **Pipeline Module (beta)** – detect bias in raw data and apply mitigation via scikit-learn–compatible transformers (e.g., instance reweighting, disparate-impact repair). Includes a typed YAML config and orchestration to build/run pipelines end-to-end.



## Features
- Unified `FairnessAnalyzer` with adapters (Fairlearn/Aequitas)
- Classification & regression fairness metrics, intersectional analysis
- Bootstrap (and small‑n Bayesian) intervals for robust statistical validation
- Effect size computation (risk ratio, Cohen’s d) for practical disparity magnitude
- **MLOps integration**: MLflow logging, pytest assertions, and CLI (`fairpipe validate`)
- Modular architecture with a user-friendly public API under `measurement/`


## Quick start: Installation

```console
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
pytest -q
```



## Measurement Module Overview

The Measurement Module is the public entry point of the toolkit.
It unifies all fairness-related functionality — metrics, statistical validation, reporting, and integration — under a single import path.

### Example Usage


```python
from fairness_pipeline_dev_toolkit.measurement import (
    FairnessAnalyzer,
    bootstrap_ci,
    beta_binomial_interval,
    risk_ratio,
    to_markdown_report,
)

# 1. Initialize the analyzer
fa = FairnessAnalyzer(min_group_size=30, backend="native")

# 2. Compute fairness metrics
res_dp = fa.demographic_parity_difference(y_pred, sensitive)
res_eq = fa.equalized_odds_difference(y_true, y_pred, sensitive)

# 3. Compute confidence intervals
ci = bootstrap_ci(y_pred, np.mean, B=1000, level=0.95)

# 4. Generate Markdown report
md = to_markdown_report({"dp": res_dp, "eo": res_eq})
print(md)
```

## Demo and Sytem Tests
Run the full end-to-end system tests:

```console
pytest tests/system -q
```

Generate a quick sample fairness report with CI and effect sizes:

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

## Project Layout

```markdown
fairness_pipeline_dev_toolkit/
├── cli/              # CLI and command-line interface (fairpipe)
├── metrics/          # Adapters + core metric definitions
├── stats/            # Bootstrap, Bayesian, effect size utilities
├── integration/      # MLflow, pytest, and reporting integrations
├── utils/            # Validation, intersectional utilities
└── measurement/      # Unified public API (Measurement Module)
```

## Documentation

See `docs/ADR-001-architecture.md` for design decisions.
The API is stable as of version `v0.1.0`.


## Pipeline Module (beta)

The Pipeline Module gives data engineers a **standardized, configurable, and automated** way to detect and mitigate bias **before** modeling.

### What it includes (Phase 1–2)
- **Bias Detection Engine**
  - Representation bias (counts/ratios vs. optional benchmarks)
  - Statistical disparity across groups (chi-square/ANOVA selection by type)
  - Proxy variable detection (Cramér’s V / η² with thresholding)
- **Transformers (sklearn-compatible)**
  - `InstanceReweighting` – computes per-row `sample_weight` to correct representation disparities
  - `DisparateImpactRemover` – rank-preserving repair for continuous features wrt a sensitive attribute
- **Config + Orchestration**
  - Typed YAML config (`PipelineConfig`, `PipelineStep`)
  - Build and run via `build_pipeline(...)`, `apply_pipeline(...)`, `run_detectors(...)`
- **CLI**
  - `fairpipe pipeline-run` to run detectors + transform data from a config

### Directory (new/updated)

```text
fairness_pipeline_dev_toolkit/
└── pipeline/
  ├── __init__.py
  ├── config/
  │   ├── __init__.py
  │   └── schema.py               # PipelineConfig, PipelineStep, loader
  ├── detectors/
  │   ├── __init__.py
  │   ├── representation.py       # RepresentationBiasDetector
  │   ├── disparity.py            # StatisticalDisparityDetector
  │   └── proxy.py                # ProxyVariableDetector
  ├── orchestration/
  │   ├── __init__.py
  │   └── engine.py               # run_detectors, build_pipeline, apply_pipeline
  └── transformers/
    ├── __init__.py
    ├── instance_reweighting.py
    └── disparate_impact.py
```

### Minimal config example
```yaml
# config/pipeline.dev.yml
sensitive: ["group"]
alpha: 0.05
proxy_threshold: 0.30
report_out: "artifacts/pipeline_bias_report.json"

pipeline:
  - name: reweight
    transformer: InstanceReweighting
    params:
      strategy: "target"         # or "uniform"
      benchmarks:
        group: {A: 0.5, B: 0.5}  # optional

  - name: di_repair
    transformer: DisparateImpactRemover
    params:
      features: ["income", "score"]
      sensitive: "group"
      repair_level: 0.8
```

### CLI Usage

# Validate env (one-time)
```console
pip install -e .
```

# Run detectors and pipeline from a config
```console
 python -m fairness_pipeline_dev_toolkit.cli.main pipeline \
  --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
  --csv dev_sample.csv \
  --out-csv artifacts/sample.transformed.csv \
  --detector-json artifacts/detectors.json \
  --report-md artifacts/pipeline_run.md
```

## Python API Usage


```python
import pandas as pd
from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    run_detectors, build_pipeline, apply_pipeline
)

cfg = load_config("pipeline.config.yml")
df = pd.read_csv("dev_sample.csv")

# 1) Detect
report = run_detectors(df, cfg)        # dict JSON-safe

# 2) Build & apply
pipe = build_pipeline(cfg)
Xt, artifacts = apply_pipeline(pipe, df)  # artifacts may include 'sample_weight'
```

















