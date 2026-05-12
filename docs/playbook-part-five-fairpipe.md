# Part Five · The fairpipe toolchain (as implemented)

**fairpipe** is the PyPI package name for this toolkit. It is published by **Svrus LLC** under the **Apache-2.0** license and requires **Python ≥ 3.10**. The installable artifact exposes two import roots on the same installation: **`fairpipe.*`** (compatibility shims) and **`fairness_pipeline_dev_toolkit.*`** (full source layout). This part uses **`fairpipe.*`** where shims exist, and **`fairness_pipeline_dev_toolkit.*`** only where no `fairpipe` submodule exists (notably the optional REST API and internal stats helpers).

The sections below are **reference + runnable examples**. Adjust paths and column names to your data. For the exact release number, see `pyproject.toml` or `fairpipe.__version__`.

---

## §0 · Package layout and installation

### Install

```bash
pip install fairpipe
```

### Optional extras (from `pyproject.toml`)

| Extra | Purpose |
|--------|---------|
| `training` | PyTorch, plotting (for `FairnessRegularizerLoss`, `LagrangianFairnessTrainer`, Pareto utilities) |
| `monitoring` | Streamlit, Dash, Plotly, Jinja2, PyWavelets (dashboards / drift tooling) |
| `adapters` | Fairlearn (and Aequitas on supported Python versions) for metric backends and `ReductionsWrapper` |
| `api` | FastAPI stack for `fairpipe serve` |
| `dev` | Test and lint tooling only |

```bash
pip install 'fairpipe[adapters]'
pip install 'fairpipe[training]'
pip install 'fairpipe[monitoring]'
pip install 'fairpipe[api]'
```

### Shim modules under `fairpipe`

`fairpipe.metrics`, **`fairpipe.measurement`** (facade mirroring `fairness_pipeline_dev_toolkit.measurement`), `fairpipe.pipeline`, `fairpipe.training`, `fairpipe.monitoring`, `fairpipe.integration`, `fairpipe.exceptions`.

The **root** package re-exports: `FairnessAnalyzer`, `MetricResult`, `to_markdown_report`, `log_fairness_metrics`, `assert_fairness`, plus **`load_data`** from I/O.

**Import paths (measurement-facing):** any of the following are valid and refer to the same underlying types where applicable:

```python
from fairpipe import FairnessAnalyzer, MetricResult, load_data, to_markdown_report, assert_fairness
from fairpipe.metrics import FairnessAnalyzer, MetricResult
from fairpipe.measurement import FairnessAnalyzer, assert_fairness, to_markdown_report  # facade + stats helpers when available
from fairpipe.pipeline import load_config, build_pipeline, apply_pipeline, run_detectors
from fairpipe.integration import execute_workflow, log_fairness_metrics
```

### Data loading

```python
from fairpipe import load_data

df = load_data("data/holdout.csv")  # or .parquet / .pq
```

(`fairness_pipeline_dev_toolkit.io.load_data` is the same function.)

---

## §1 · Metrics (measurement)

### Primary types

- **`fairpipe.metrics.FairnessAnalyzer`** — orchestrates fairness metrics with optional intersectional grouping, `min_group_size`, and optional bootstrap confidence intervals.
- **`fairpipe.metrics.FairnessAnalyzerDataFrameProxy`** — returned by `FairnessAnalyzer.from_dataframe(...)`; binds column names so you call metric methods without passing arrays each time.
- **`fairpipe.metrics.MetricResult`** — dataclass: `metric`, `value`, optional `ci`, `effect_size`, `n_per_group`.

### Backends

Constructor argument `backend=None` (auto-select), or **`"native"`**, **`"fairlearn"`**, **`"aequitas"`**. Fairlearn/Aequitas require **`pip install fairpipe[adapters]`** where applicable.

### Methods on `FairnessAnalyzer` (and the dataframe proxy, where columns allow)

- **`demographic_parity_difference(y_pred, sensitive, …)`** — optional intersectional mode via `intersectional=True` with `attrs_df` / `columns`; optional bootstrap CI via `with_ci`, `ci_level`, `ci_samples`, `ci_method` (**`"percentile"`** or **`"bca"`** only), optional effect sizes.
- **`equalized_odds_difference(y_true, y_pred, sensitive, …)`** — same optional intersectional and CI/effect-size knobs; supports multi-attribute handling via `attrs_df` / `columns`.
- **`mae_parity_difference(y_true, y_pred, sensitive, …)`** — for score-based regression-style parity; same CI options.

### Code · Array API (single sensitive column)

```python
from fairpipe import FairnessAnalyzer

analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

dp = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True,
    ci_level=0.95,
    ci_samples=1000,
    ci_method="percentile",  # or "bca"
    with_effect_size=True,
)
print(dp.value, dp.ci, dp.n_per_group)

eo = analyzer.equalized_odds_difference(
    y_true=y_true,
    y_pred=y_pred,
    sensitive=sensitive,
    with_ci=True,
    ci_method="bca",
)
```

### Code · DataFrame-bound API

```python
from fairpipe import FairnessAnalyzer

proxy = FairnessAnalyzer.from_dataframe(
    df,
    y_pred_col="y_pred",
    sensitive_col="gender",
    y_true_col="y_true",
    min_group_size=30,
    backend="native",
)
dp = proxy.demographic_parity_difference(with_ci=True)
eo = proxy.equalized_odds_difference(with_ci=True)
```

### Code · Intersectional cohorts (multiple attributes)

```python
from fairpipe import FairnessAnalyzer

analyzer = FairnessAnalyzer(min_group_size=50, backend="native")
attrs = df[["gender", "region"]]

eo = analyzer.equalized_odds_difference(
    y_true=df["y_true"].to_numpy(),
    y_pred=df["y_pred"].to_numpy(),
    sensitive=df["gender"].to_numpy(),  # unused when intersectional=True with attrs_df
    intersectional=True,
    attrs_df=attrs,
    columns=["gender", "region"],
    with_ci=True,
)
```

### Reporting and tests

- **`fairpipe.integration.to_markdown_report(results)`** — builds a Markdown table from a mapping of metric names to `MetricResult`-like objects or dicts.
- **`fairpipe.integration.assert_fairness(value, threshold, comparator="<=", allow_nan=False, context=None)`** — pytest-style helper; raises **`AssertionError`** on breach (not a dedicated fairness exception type).

```python
from fairpipe import FairnessAnalyzer, to_markdown_report, assert_fairness

analyzer = FairnessAnalyzer(min_group_size=30)
results = {
    "demographic_parity_difference": analyzer.demographic_parity_difference(
        y_pred=y_pred, sensitive=sensitive, with_ci=True
    ),
    "equalized_odds_difference": analyzer.equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive=sensitive, with_ci=True
    ),
}
print(to_markdown_report(results, title="Audit"))

assert_fairness(
    results["demographic_parity_difference"].value,
    threshold=0.10,
    comparator="<=",
    context="demographic_parity_difference",
)
```

### Lower-level statistics (same install, not under `fairpipe.*`)

Examples: **`fairness_pipeline_dev_toolkit.stats.bootstrap.bootstrap_ci`** (`method` **`"percentile"`** or **`"bca"`**), **`fairness_pipeline_dev_toolkit.stats.multipletests.benjamini_hochberg`**. These are ordinary helpers; they are **not** wired into a single `Metrics.compute`-style entry point.

```python
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci
from fairness_pipeline_dev_toolkit.stats.multipletests import benjamini_hochberg
import numpy as np

x = np.random.randn(80)
low, high = bootstrap_ci(x, lambda v: float(np.mean(v)), B=500, level=0.95, method="bca")

adj, order = benjamini_hochberg([0.01, 0.04, 0.10])
```

---

## §2 · Pipeline configuration and transforms

### Loading config

- **`fairpipe.pipeline.load_config(path=..., text=..., obj=..., profile=...)`** → **`PipelineConfig`**.
- If `path` is omitted and `text`/`obj` are not given, discovery order is: **`.fairpipe/config.yml`** (walk upward from cwd), then **`~/.fairpipe/config.yml`**, else error.
- YAML may define a top-level **`profiles`** map. Active profile: `profile` argument, or env **`FPDT_PROFILE`**, or test auto-detection, or the sole profile if only one exists. Selected profile is **shallow-merged** over top-level keys.

Full validation rules: `fairness_pipeline_dev_toolkit/pipeline/config/loader.py`.

### `PipelineConfig` fields (validated subset)

| Field | Role |
|--------|------|
| `sensitive` | List of sensitive attribute column names (required). |
| `benchmarks` | Optional dict-of-dicts for detector benchmarks. |
| `alpha` | Float, default `0.05`. |
| `proxy_threshold` | Float, default `0.30`. |
| `report_out` | Optional path string for detector report output. |
| `pipeline` | List of steps (legacy key **`steps`** also accepted). Each step: **`name`**, **`transformer`** (string), **`params`** (dict). |
| `training` | Optional: **`method`** ∈ `reductions` \| `regularized` \| `lagrangian`, **`target_column`**, **`params`**. |
| `fairness_metric` | Optional string (e.g. disparity metric name for workflow validation). |
| `validation_threshold` | Optional float cap for that metric in the integrated workflow. |

### Registered transformer names (strings in YAML)

Map to sklearn-style components, including:

- **`InstanceReweighting`**
- **`DisparateImpactRemover`**
- **`ReweighingTransformer`**
- **`ProxyDropper`**

### Orchestration API

- **`fairpipe.pipeline.build_pipeline(config)`** — build sklearn `Pipeline` from config.
- **`fairpipe.pipeline.apply_pipeline(pipe, df)`** — apply to a `DataFrame`.
- **`fairpipe.pipeline.run_detectors(df, cfg)`** — run bias detectors; returns a report object with JSON serialization where implemented.

### Code · Pipeline-only YAML and Python

```yaml
# example-pipeline.yml
sensitive: ["sensitive"]

pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
  - name: di
    transformer: "DisparateImpactRemover"
    params:
      sensitive: "sensitive"
      features: ["score"]
      repair_level: 0.8

alpha: 0.05
proxy_threshold: 0.30
```

```python
from fairpipe import load_data
from fairpipe.pipeline import load_config, build_pipeline, apply_pipeline, run_detectors

cfg = load_config("example-pipeline.yml")
df = load_data("data.csv")

report = run_detectors(df=df, cfg=cfg)
pipe = build_pipeline(cfg)
Xt, _ = apply_pipeline(pipe, df)
```

### Code · Profiles (shallow merge)

```yaml
sensitive: ["s"]
alpha: 0.05

profiles:
  staging:
    pipeline:
      - name: only_di
        transformer: "DisparateImpactRemover"
        params: { sensitive: "s", features: ["x1"], repair_level: 0.7 }
```

```python
import os

os.environ["FPDT_PROFILE"] = "staging"
cfg = load_config("profiles.yml")
```

---

## §3 · Training (optional dependencies)

### Exported from `fairpipe.training`

- **`ReductionsWrapper`** — sklearn classifier wrapping Fairlearn **`ExponentiatedGradient`**; **`fit(X, y, sensitive_features=...)`**; constructed with **`base_estimator`**, **`constraint`** (Fairlearn constraint object), **`eps`**, **`T`**, optional **`kwargs`** (forwarded; `max_iter` defaulted for the wrapper). Requires Fairlearn via **`fairpipe[adapters]`**.
- **`FairnessRegularizerLoss`** — PyTorch loss augmentation (requires **`fairpipe[training]`**).
- **`LagrangianFairnessTrainer`** — PyTorch trainer with fairness constraint **`demographic_parity`** or **`equal_opportunity`** (requires **`fairpipe[training]`**).
- **`GroupFairnessCalibrator`** — group-wise calibration helper.
- **`sweep_pareto`**, **`plot_pareto`** — Pareto sweep / plot for regularized training demos.

There is **no** separate exported class named `FairClassifier`, `FairLossWrapper`, or `Reweighing` under `fairpipe.training`; mitigation aligned with Fairlearn reductions is **`ReductionsWrapper`**. Instance reweighting / disparate impact repair are **pipeline transformers** (§2), not `fairpipe.training` exports.

### Code · Fairlearn reductions (`ReductionsWrapper`)

```python
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import EqualizedOdds
from fairpipe.training import ReductionsWrapper

model = ReductionsWrapper(
    LogisticRegression(max_iter=1000),
    EqualizedOdds(),
    eps=0.01,
    T=50,
).fit(X, y, sensitive_features=sensitive_features)

pred = model.predict(X_test)
```

### Code · Lagrangian trainer (same idea as `fairpipe train-lagrangian`)

The CLI reads CSV columns `f0..f{d-1}`, `y`, `s`. Programmatic use:

```python
import torch
import torch.nn as nn
from fairpipe.training import LagrangianFairnessTrainer

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)

X = torch.randn(500, 8)
y = (torch.randn(500) > 0).long()
s = torch.randint(0, 2, (500,))

trainer = LagrangianFairnessTrainer(
    MLP(8),
    fairness="demographic_parity",
    dp_tolerance=0.02,
    model_lr=1e-3,
    lambda_lr=1e-2,
    device="cpu",
)
history = trainer.fit(X, y, s, epochs=5, batch_size=64, verbose=True)
```

### Code · Fairness regularizer + Pareto sweep (`fairpipe train-regularized`)

```python
from fairpipe.training import sweep_pareto, plot_pareto

points = sweep_pareto(
    X_train, y_train, s_train,
    X_val, y_val, s_val,
    etas=[0.0, 0.2, 0.5],
    epochs=10,
    lr=1e-3,
    device="cpu",
)
plot_pareto(points, save_path="pareto.png")
```

### Code · Group calibration

CLI expects columns **`score`**, **`y`**, **`g`**:

```bash
fairpipe calibrate --csv scores.csv --method platt --out-csv calibrated.csv
```

```python
import numpy as np
from fairpipe.training import GroupFairnessCalibrator

scores = np.random.rand(200)
labels = (scores + np.random.randn(200) * 0.2 > 0.5).astype(int)
groups = np.array([0, 1] * 100)

cal = GroupFairnessCalibrator(method="platt", min_samples=20).fit(scores, labels, groups)
calibrated = cal.transform(scores, groups)
```

---

## §4 · Monitoring (optional dependencies)

### Exported from `fairpipe.monitoring`

- **`RealTimeFairnessTracker`**, **`TrackerConfig`**, **`ColumnMap`**
- **`FairnessDriftAndAlertEngine`**, **`DriftConfig`**, **`MonitoringSettings`**, **`AlertEvent`**
- **`FairnessReportingDashboard`**, **`ReportConfig`**
- **`FairnessABTestAnalyzer`**

**`FairnessDriftAndAlertEngine`** compares recent vs reference windows (e.g. KS-based scoring on metric time series), optionally with wavelet decomposition when PyWavelets is available and configured. This module does **not** implement external incident integrations (PagerDuty, Slack, etc.) in code; it emits in-memory **`AlertEvent`** records for downstream use.

### Code · Drift engine

Requires monitoring-related dependencies for dashboards; the engine uses `pandas`, `numpy`, `scipy`, and optionally PyWavelets for multi-scale mode.

```python
import pandas as pd
from fairpipe.monitoring import FairnessDriftAndAlertEngine, DriftConfig, MonitoringSettings

rng = pd.date_range("2025-01-01", periods=80, freq="D")
rows = []
for i, t in enumerate(rng):
    rows.append(
        {
            "timestamp": t,
            "metric": "DP_global",
            "group_key": "all",
            "value": 0.05 + 0.001 * i,
            "n": 1000,
        }
    )
metrics_ts = pd.DataFrame(rows)

engine = FairnessDriftAndAlertEngine(MonitoringSettings(drift=DriftConfig()))
alerts = engine.analyze(metrics_ts, window_points=12, ref_points=48)
for a in alerts:
    print(a.metric, a.severity, a.reason)
```

---

## §5 · Integration and workflow

### Exported from `fairpipe.integration`

- **`execute_workflow(config, df, output_dir=None, min_group_size=30, train_size=0.8)`** → **`WorkflowResult`**  
  Runs the **three-step** flow: baseline measurement (when **`training`** + **`fairness_metric`** are set), transform + train, final validation against **`validation_threshold`**. Returns metrics dicts, **`ValidationResult`**, fitted **`model`**, **`transformed_df`**, **`predictions`**, and **`artifacts`**.
- **`WorkflowResult`**, **`ValidationResult`**
- **`log_workflow_results`** — MLflow-oriented logging helper when MLflow is active.
- **`to_markdown_report`**, **`assert_fairness`** — as in §1.

There is **no** `Governance` class, FDR generator, TRS calculator, evidence-pack exporter, or GRC connectors in this package.

Feature columns for the workflow are **all columns except** the target and the listed sensitive columns.

### Code · `execute_workflow` and CLI

```yaml
# workflow.yml
sensitive: ["s"]
pipeline: []

training:
  method: reductions
  target_column: "y"
  params:
    constraint: demographic_parity
    eps: 0.02
    T: 30

fairness_metric: "demographic_parity_difference"
validation_threshold: 0.15
```

```python
from fairpipe import load_data
from fairpipe.pipeline import load_config
from fairpipe.integration import execute_workflow

df = load_data("train.csv")  # must include feature columns, "y", "s"
cfg = load_config("workflow.yml")
result = execute_workflow(cfg, df, output_dir="artifacts/run1", min_group_size=30)

print(result.validation_result.passed, result.validation_result.message)
print(result.final_metrics)
```

```bash
export FAIRPIPE_CONFIG_PATH=workflow.yml
fairpipe run-pipeline --csv train.csv --output-dir artifacts/run1
```

### Code · MLflow logging

```python
from fairpipe.integration import log_fairness_metrics, to_markdown_report

results = {"demographic_parity_difference": dp_result}  # MetricResult or dict-like
log_fairness_metrics(
    results,
    prefix="fairness_",
    artifact_name="fairness_report.md",
    artifact_content=to_markdown_report(results),
)
```

`fairpipe run-pipeline` can log via **`--mlflow-experiment`** / **`--mlflow-run-name`** when MLflow is configured.

---

## §6 · Exceptions

**`fairpipe.exceptions`** (hierarchy under **`FairnessToolkitError`**):  
**`ConfigValidationError`**, **`MetricComputationError`**, **`PipelineExecutionError`**, **`TrainingError`**, **`DataValidationError`**, **`DependencyError`**.

```python
from fairpipe.exceptions import ConfigValidationError, FairnessToolkitError
```

---

## §7 · Command-line interface

**Entry point:** `fairpipe` → `fairness_pipeline_dev_toolkit.cli.main:main`.

| Command | Purpose |
|---------|--------|
| `fairpipe version` | Print package version string. |
| `fairpipe validate` | Load CSV/Parquet via `load_data`; compute disparity metrics; print Markdown report; optional `--out` file. |
| `fairpipe sample-check` | Placeholder: checks for `dev_sample.csv` in cwd; non-blocking. |
| `fairpipe pipeline` | Load YAML config (`--config` or **`FAIRPIPE_CONFIG_PATH`**); run detectors unless `--no-detectors`; build and apply pipeline; optional `--out-csv`, `--detector-json`, `--report-md`. Optional `--profile` for multi-profile YAML. |
| `fairpipe train-regularized` | Demo sweep with `sweep_pareto` on CSV with columns `f*`, `y`, `s`; writes JSON (and optional PNG). |
| `fairpipe train-lagrangian` | Train with `LagrangianFairnessTrainer`; `--fairness` `demographic_parity` or `equal_opportunity`; writes JSON. |
| `fairpipe calibrate` | Group calibration: CSV cols `score`, `y`, `g`; `--method` `platt` or `isotonic`. |
| `fairpipe run-pipeline` | Full `execute_workflow`; `--config` or **`FAIRPIPE_CONFIG_PATH`**; `--csv`; `--output-dir`; `--min-group-size`; `--train-size`; optional **`--mlflow-experiment`** / **`--mlflow-run-name`**. Exit code **1** if validation fails or an exception occurs. |
| `fairpipe serve` | Start HTTP API (requires **`fairpipe[api]`**); `--host`, `--port`, `--reload`, `--workers`. |

### Common environment variables

| Variable | Role |
|----------|------|
| **`FAIRPIPE_CONFIG_PATH`** | Default config path for `pipeline` / `run-pipeline` when `--config` omitted. |
| **`FAIRPIPE_MIN_GROUP_SIZE`** | Default for `validate` / `run-pipeline` minimum group size. |
| **`FAIRPIPE_MLFLOW_EXPERIMENT`** | Default MLflow experiment for `run-pipeline`. |
| **`FAIRPIPE_LOG_LEVEL`**, **`FAIRPIPE_LOG_FILE`**, **`FAIRPIPE_JSON_LOGS`** | Logging for the CLI. |
| **`FPDT_PROFILE`** | Default YAML profile name when configs use `profiles`. |

**`validate` flags (representative):** `--csv`, `--y-true`, exactly one of `--y-pred` or `--score`, `--sensitive` (one or more columns), `--min-group-size`, `--backend` (`auto|native|fairlearn|aequitas`), `--out`, `--with-ci`, `--ci-level`, `--bootstrap-B`, `--with-effects`.

**Not in the current CLI:** `fairpipe init`, `fairpipe config validate`, nested `fairpipe pipeline run`, `fairpipe audit *`, `fairpipe recipes *`, `fairpipe doctor`.

### Shell examples

```bash
fairpipe version

fairpipe validate \
  --csv data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --with-ci --out report.md

fairpipe pipeline --config pipeline.config.yml --csv data.csv --out-csv out.csv

fairpipe run-pipeline --config workflow.yml --csv data.csv --output-dir artifacts/

fairpipe train-regularized --csv nn_demo.csv --out-json pareto.json --out-png pareto.png

fairpipe train-lagrangian --csv nn_demo.csv --fairness demographic_parity --out-json lag.json
```

---

## §8 · Optional REST API (`fairpipe[api]`)

**Import:** `from fairness_pipeline_dev_toolkit.api import create_app, ResultStore`  
(There is **no** `fairpipe.api` shim; the CLI imports **`create_app`** from **`fairness_pipeline_dev_toolkit.api.app`**.)

**`create_app()`** returns a FastAPI app with routers for **health**, **validate**, **pipeline**, and **workflow** (`fairness_pipeline_dev_toolkit/api/routes/`). **`fairpipe serve`** runs this app via Uvicorn; with **`--reload`**, the import string is **`fairness_pipeline_dev_toolkit.api.app:create_app`** with **`factory=True`**.

```bash
pip install 'fairpipe[api]'
fairpipe serve --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/docs
```

```python
from fairness_pipeline_dev_toolkit.api import create_app

app = create_app()
```

---

## §9 · Minimal configuration example (matches `PipelineConfig`)

File name is arbitrary; **`pipeline.config.yml`** and **`.fairpipe/config.yml`** are common. Optional top-level **`profiles`** can supply alternate `pipeline` / `training` blocks per profile; merge rules are **shallow** over the base keys.

```yaml
sensitive: ["sensitive"]

benchmarks:
  sensitive:
    A: 0.40
    B: 0.40
    C: 0.20

alpha: 0.05
proxy_threshold: 0.30
report_out: "artifacts/pipeline_bias_report.json"

pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
  - name: di
    transformer: "DisparateImpactRemover"
    params:
      sensitive: "sensitive"
      features: ["score"]
      repair_level: 0.8

# Optional: integrated workflow training (used by execute_workflow / run-pipeline)
training:
  method: reductions # or: regularized | lagrangian
  target_column: "y"
  params: {}

fairness_metric: "demographic_parity_difference" # example; must match workflow logic
validation_threshold: 0.10
```

---

## §10 · CI usage (accurate pattern)

A minimal GitHub Actions job can:

1. **`pip install fairpipe`** (and extras as needed).
2. Run **`fairpipe validate`** on a CSV with the required columns, or **`fairpipe pipeline --config … --csv …`**, or **`fairpipe run-pipeline`** when a full workflow config exists.
3. Use **`assert_fairness`** in pytest on metric values your code reads from **`FairnessAnalyzer`** or workflow results.

The repository documents a separate composite action (**[SvrusIO/fairpipe-action](https://github.com/SvrusIO/fairpipe-action)**) for validation in CI; that is **external** to the Python package itself.

---

— **End of Part Five (as implemented)** —
