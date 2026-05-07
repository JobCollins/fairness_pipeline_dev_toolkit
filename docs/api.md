# API Reference

Complete API documentation for the Fairness Pipeline Development Toolkit.

> **Namespace note:** All APIs documented here are available under both the `fairpipe.*` and
> `fairness_pipeline_dev_toolkit.*` namespaces. The `fairpipe` package is a thin re-export shim —
> object identity is fully preserved (`fairpipe.metrics.FairnessAnalyzer is
> fairness_pipeline_dev_toolkit.metrics.FairnessAnalyzer` evaluates to `True`). Either namespace
> may be used interchangeably.

---

## Table of Contents

- [Core Metrics](#core-metrics)
- [I/O Utilities](#io-utilities)
- [Pipeline Utilities](#pipeline-utilities)
- [Integration & Workflow](#integration--workflow)
- [REST API](#rest-api)
- [Training](#training)
- [Monitoring](#monitoring)
- [Exceptions](#exceptions)
- [Statistical Utilities](#statistical-utilities)

---

## Core Metrics

### `FairnessAnalyzer`

Main class for computing fairness metrics with statistical validation.

**Location:** `fairness_pipeline_dev_toolkit.metrics.FairnessAnalyzer`

**Constructor:**

```python
FairnessAnalyzer(
    *,
    min_group_size: int = 30,
    nan_policy: str = "exclude",
    backend: Optional[str] = None
)
```

**Parameters:**
- `min_group_size` (int): Minimum number of samples required per group (default: 30)
- `nan_policy` (str): How to handle NaN values in sensitive attributes. Options: `"exclude"` (default), `"include"`
- `backend` (str, optional): Backend adapter to use. Options: `"native"`, `"fairlearn"`, `"aequitas"`, or `None` (auto-select)

**Properties:**
- `backend` (str): The currently active backend adapter name

**Class Methods:**

#### `from_dataframe()`

Create a column-bound proxy from a DataFrame so that metric methods need no column arguments per call.

```python
@classmethod
def from_dataframe(
    cls,
    df: pd.DataFrame,
    y_pred_col: str,
    sensitive_col: str,
    y_true_col: str | None = None,
    y_score_col: str | None = None,
    min_group_size: int = 30,
    backend: str = "native",
) -> FairnessAnalyzerDataFrameProxy
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `y_pred_col` (str): Column name for predictions
- `sensitive_col` (str): Column name for the sensitive attribute
- `y_true_col` (str, optional): Column name for ground-truth labels (required for EOD/MAE methods)
- `y_score_col` (str, optional): Column name for prediction scores
- `min_group_size` (int): Minimum group size (default: 30)
- `backend` (str): Backend adapter (default: `"native"`)

**Raises:** `KeyError` if any specified column is not present in `df`:
`"Column '{col}' not found in DataFrame. Available columns: [...]"`

**Example:**
```python
from fairpipe.metrics import FairnessAnalyzer

proxy = FairnessAnalyzer.from_dataframe(
    df,
    y_pred_col="y_pred",
    sensitive_col="gender",
    y_true_col="y_true",
)
result = proxy.demographic_parity_difference(with_ci=True)
result_eod = proxy.equalized_odds_difference()
```

---

**Methods:**

#### `demographic_parity_difference()`

Compute the demographic parity difference (DPD) metric.

```python
def demographic_parity_difference(
    y_pred: np.ndarray | pd.Series | list,
    sensitive: np.ndarray | pd.Series | list,
    *,
    intersectional: bool = False,
    attrs_df: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    with_ci: bool = True,
    ci_level: float = 0.95,
    ci_method: str = "percentile",
    ci_samples: int = 1000,
    with_effect_size: bool = True
) -> Result
```

**Parameters:**
- `y_pred` (np.ndarray | pd.Series | list): Binary predictions (0/1) or continuous scores
- `sensitive` (np.ndarray | pd.Series | list): Sensitive attribute values
- `intersectional` (bool): If True, compute intersectional fairness across multiple attributes
- `attrs_df` (pd.DataFrame, optional): Required if `intersectional=True`. DataFrame containing all sensitive attributes
- `columns` (List[str], optional): Column names in `attrs_df` to use for intersectional analysis
- `with_ci` (bool): Compute bootstrap confidence intervals (default: True)
- `ci_level` (float): Confidence level for intervals (default: 0.95)
- `ci_method` (str): Bootstrap method. Options: `"percentile"` (default), `"bca"`
- `ci_samples` (int): Number of bootstrap samples (default: 1000)
- `with_effect_size` (bool): Compute effect size (risk ratio) (default: True)

**Returns:** `Result` object with:
- `metric` (str): Metric name
- `value` (float): Point estimate of DPD
- `ci` (tuple[float, float] | None): Confidence interval
- `effect_size` (float | None): Risk ratio effect size
- `n_per_group` (Dict[str, int] | None): Sample sizes per group

**Example:**
```python
from fairpipe.metrics import FairnessAnalyzer
import numpy as np

analyzer = FairnessAnalyzer(min_group_size=30)
result = analyzer.demographic_parity_difference(
    y_pred=y_pred,
    sensitive=gender,
    with_ci=True,
    ci_level=0.95
)
print(f"DPD: {result.value:.4f}")
print(f"95% CI: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
```

#### `equalized_odds_difference()`

Compute the equalized odds difference (EOD) metric.

```python
def equalized_odds_difference(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    sensitive: np.ndarray | pd.Series | list,
    *,
    intersectional: bool = False,
    attrs_df: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    with_ci: bool = True,
    ci_level: float = 0.95,
    ci_method: str = "percentile",
    ci_samples: int = 1000,
    with_effect_size: bool = True
) -> Result
```

**Parameters:**
- `y_true` (np.ndarray | pd.Series | list): Ground truth binary labels (0/1)
- `y_pred` (np.ndarray | pd.Series | list): Binary predictions (0/1)
- `sensitive` (np.ndarray | pd.Series | list): Sensitive attribute values
- `intersectional` (bool): If True, compute intersectional fairness
- `attrs_df` (pd.DataFrame, optional): Required if `intersectional=True`
- `columns` (List[str], optional): Column names for intersectional analysis
- `with_ci` (bool): Compute bootstrap confidence intervals (default: True)
- `ci_level` (float): Confidence level (default: 0.95)
- `ci_method` (str): Bootstrap method (default: "percentile")
- `ci_samples` (int): Number of bootstrap samples (default: 1000)
- `with_effect_size` (bool): Compute effect size (default: True)

**Returns:** `Result` object with EOD metric value, CI, and effect size.

**Example:**
```python
result = analyzer.equalized_odds_difference(
    y_true=y_true,
    y_pred=y_pred,
    sensitive=gender,
    with_ci=True
)
```

#### `mae_parity_difference()`

Compute the mean absolute error (MAE) parity difference for regression tasks.

```python
def mae_parity_difference(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    sensitive: np.ndarray | pd.Series | list,
    *,
    intersectional: bool = False,
    attrs_df: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    with_ci: bool = True,
    ci_level: float = 0.95,
    ci_method: str = "percentile",
    ci_samples: int = 1000,
    with_effect_size: bool = True
) -> Result
```

**Parameters:**
- `y_true` (np.ndarray | pd.Series | list): Ground truth continuous values
- `y_pred` (np.ndarray | pd.Series | list): Predicted continuous values
- `sensitive` (np.ndarray | pd.Series | list): Sensitive attribute values
- `intersectional` (bool): If True, compute intersectional fairness
- `attrs_df` (pd.DataFrame, optional): Required if `intersectional=True`
- `columns` (List[str], optional): Column names for intersectional analysis
- `with_ci` (bool): Compute bootstrap confidence intervals (default: True)
- `ci_level` (float): Confidence level (default: 0.95)
- `ci_method` (str): Bootstrap method (default: "percentile")
- `ci_samples` (int): Number of bootstrap samples (default: 1000)
- `with_effect_size` (bool): Compute effect size (Cohen's d) (default: True)

**Returns:** `Result` object with MAE parity difference, CI, and effect size.

**Example:**
```python
result = analyzer.mae_parity_difference(
    y_true=y_true,
    y_pred=y_pred,
    sensitive=race,
    with_ci=True
)
```

### `MetricResult`

Result object returned by all metric computations.

**Location:** `fairness_pipeline_dev_toolkit.metrics.MetricResult`

**Attributes:**
- `metric` (str): Name of the metric (e.g., "demographic_parity_difference")
- `value` (float): Point estimate of the metric
- `ci` (tuple[float, float] | None): Confidence interval [lower, upper]
- `effect_size` (float | None): Effect size (risk ratio, Cohen's d, etc.)
- `n_per_group` (Dict[str, int] | None): Sample sizes per group

**Example:**
```python
from fairness_pipeline_dev_toolkit.metrics import MetricResult

result = MetricResult(
    metric="demographic_parity_difference",
    value=0.15,
    ci=(0.10, 0.20),
    effect_size=1.5,
    n_per_group={"M": 500, "F": 500}
)
```

### `FairnessAnalyzerDataFrameProxy`

Column-bound proxy returned by `FairnessAnalyzer.from_dataframe()`. Stores a DataFrame and column names so metric methods can be called without repeating column arguments.

**Location:** `fairness_pipeline_dev_toolkit.metrics.FairnessAnalyzerDataFrameProxy`

**Methods:** exposes the same three metric methods as `FairnessAnalyzer` — `demographic_parity_difference(**kwargs)`, `equalized_odds_difference(**kwargs)`, `mae_parity_difference(**kwargs)` — forwarding all keyword arguments to the underlying analyzer.

**Example:**
```python
from fairpipe.metrics import FairnessAnalyzer
import pandas as pd

df = pd.read_csv("predictions.csv")
proxy = FairnessAnalyzer.from_dataframe(
    df,
    y_pred_col="y_pred",
    sensitive_col="gender",
    y_true_col="y_true",
    min_group_size=30,
)

dpd = proxy.demographic_parity_difference(with_ci=True)
eod = proxy.equalized_odds_difference(with_ci=False)
mae = proxy.mae_parity_difference()
```

---

## I/O Utilities

### `load_data()`

Load a tabular data file into a DataFrame with automatic format detection.

**Location:** `fairness_pipeline_dev_toolkit.io.load_data` (also `fairpipe.load_data`)

```python
def load_data(path: str | Path) -> pd.DataFrame
```

**Supported formats:** `.csv`, `.parquet`, `.pq` — detected automatically from the file extension.

**Parameters:**
- `path` (str | Path): Path to the data file

**Returns:** `pd.DataFrame`

**Raises:**
- `FileNotFoundError`: `"File not found: {path}"`
- `ValueError`: `"Unsupported file format '{ext}'. Supported: .csv, .parquet, .pq"`

**Example:**
```python
from fairpipe.io import load_data

df_csv     = load_data("data.csv")
df_parquet = load_data("data.parquet")
df_pq      = load_data("data.pq")
```

All CLI commands that accept `--csv` use `load_data()` internally, so `.parquet` and `.pq` paths work transparently:

```bash
fairpipe validate --csv data.parquet --y-true y_true --y-pred y_pred --sensitive gender
fairpipe pipeline --config pipeline.config.yml --csv data.parquet --out-csv output.csv
```

---

## Pipeline Utilities

### Configuration

#### `PipelineConfig`

Configuration dataclass for pipeline operations.

**Location:** `fairness_pipeline_dev_toolkit.pipeline.config.PipelineConfig`

**Attributes:**
- `sensitive` (List[str]): List of sensitive attribute column names
- `pipeline` (List[PipelineStep]): List of pipeline transformation steps
- `training` (TrainingConfig | None): Training configuration (optional)
- `benchmarks` (Dict[str, Dict[str, float]] | None): Benchmark distributions for sensitive attributes
- `alpha` (float): Significance level for statistical tests (default: 0.05)
- `proxy_threshold` (float): Correlation threshold for proxy detection (default: 0.30)

#### `load_config()`

Load pipeline configuration from YAML file.

```python
def load_config(
    path: str | Path,
    profile: Optional[str] = None
) -> PipelineConfig
```

**Parameters:**
- `path` (str | Path): Path to YAML configuration file
- `profile` (str, optional): Profile name to use (if YAML contains profiles)

**Returns:** `PipelineConfig` object

**Example:**
```python
from fairness_pipeline_dev_toolkit.pipeline import load_config

config = load_config("pipeline.config.yml")
config = load_config("config.yml", profile="training")
```

#### `find_config_file()`

Find configuration file using environment variables or default locations.

```python
def find_config_file(
    default_name: str = "config.yml"
) -> Path | None
```

**Parameters:**
- `default_name` (str): Default filename to search for (default: "config.yml")

**Returns:** `Path` to config file if found, `None` otherwise

**Example:**
```python
from fairness_pipeline_dev_toolkit.pipeline.config import find_config_file

config_path = find_config_file("pipeline.config.yml")
if config_path:
    config = load_config(config_path)
```

### Pipeline Operations

#### `build_pipeline()`

Build a transformation pipeline from configuration.

```python
def build_pipeline(
    config: PipelineConfig
) -> List[Transformer]
```

**Parameters:**
- `config` (PipelineConfig): Pipeline configuration

**Returns:** List of transformer objects

**Example:**
```python
from fairness_pipeline_dev_toolkit.pipeline import build_pipeline, load_config

config = load_config("pipeline.config.yml")
pipeline = build_pipeline(config)
```

#### `apply_pipeline()`

Apply a transformation pipeline to a DataFrame.

```python
def apply_pipeline(
    pipeline: List[Transformer],
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]
```

**Parameters:**
- `pipeline` (List[Transformer]): List of transformer objects
- `df` (pd.DataFrame): Input DataFrame

**Returns:** Tuple of (transformed DataFrame, metadata dictionary)

**Example:**
```python
from fairness_pipeline_dev_toolkit.pipeline import apply_pipeline

transformed_df, metadata = apply_pipeline(pipeline, df)
```

#### `run_detectors()`

Run bias detection on a DataFrame.

```python
def run_detectors(
    df: pd.DataFrame,
    cfg: PipelineConfig
) -> BiasReport
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `cfg` (PipelineConfig): Pipeline configuration

**Returns:** `BiasReport` object containing detection results

**Example:**
```python
from fairness_pipeline_dev_toolkit.pipeline import run_detectors, load_config

config = load_config("pipeline.config.yml")
report = run_detectors(df, config)
print(report.body)
```

### Transformers

#### `InstanceReweighting`

Reweight instances to balance sensitive attribute distributions.

**Location:** `fairness_pipeline_dev_toolkit.pipeline.InstanceReweighting`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.pipeline import InstanceReweighting

transformer = InstanceReweighting(sensitive="gender")
transformed_df = transformer.fit_transform(df)
```

#### `DisparateImpactRemover`

Remove disparate impact by repairing features.

**Location:** `fairness_pipeline_dev_toolkit.pipeline.DisparateImpactRemover`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.pipeline import DisparateImpactRemover

transformer = DisparateImpactRemover(
    features=["score", "age"],
    sensitive="gender",
    repair_level=0.8
)
transformed_df = transformer.fit_transform(df)
```

#### `ReweighingTransformer`

Reweigh instances based on sensitive attribute and target label.

**Location:** `fairness_pipeline_dev_toolkit.pipeline.ReweighingTransformer`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.pipeline import ReweighingTransformer

transformer = ReweighingTransformer(sensitive="gender", target="y")
transformed_df = transformer.fit_transform(df)
```

#### `ProxyDropper`

Drop proxy variables that are highly correlated with sensitive attributes.

**Location:** `fairness_pipeline_dev_toolkit.pipeline.ProxyDropper`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.pipeline import ProxyDropper

transformer = ProxyDropper(
    sensitive="gender",
    threshold=0.30
)
transformed_df = transformer.fit_transform(df)
```

---

## Integration & Workflow

### `execute_workflow()`

Execute the complete end-to-end workflow: baseline measurement → transform+train → validation.

**Location:** `fairness_pipeline_dev_toolkit.integration.execute_workflow`

```python
def execute_workflow(
    config: PipelineConfig,
    df: pd.DataFrame,
    output_dir: str | Path = "artifacts",
    min_group_size: int = 30,
    train_size: float = 0.8,
    random_state: int = 42,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None
) -> WorkflowResult
```

**Parameters:**
- `config` (PipelineConfig): Pipeline configuration (must include `training` section)
- `df` (pd.DataFrame): Input DataFrame
- `output_dir` (str | Path): Directory to save artifacts (default: "artifacts")
- `min_group_size` (int): Minimum group size for fairness analysis (default: 30)
- `train_size` (float): Proportion of data for training (default: 0.8)
- `random_state` (int): Random seed for train/test split (default: 42)
- `mlflow_experiment` (str, optional): MLflow experiment name (enables MLflow logging)
- `mlflow_run_name` (str, optional): MLflow run name

**Returns:** `WorkflowResult` object

**Example:**
```python
from fairness_pipeline_dev_toolkit.integration import execute_workflow
from fairness_pipeline_dev_toolkit.pipeline import load_config
import pandas as pd

config = load_config("config.yml")
df = pd.read_csv("data.csv")

result = execute_workflow(
    config=config,
    df=df,
    output_dir="artifacts/workflow",
    min_group_size=30,
    mlflow_experiment="fairness_workflow"
)

if result.validation_result.passed:
    print("✅ Validation PASSED")
else:
    print("❌ Validation FAILED")
```

### `WorkflowResult`

Result object from workflow execution.

**Location:** `fairness_pipeline_dev_toolkit.integration.WorkflowResult`

**Attributes:**
- `baseline_metrics` (Dict[str, Any]): Baseline fairness metrics
- `final_metrics` (Dict[str, Any]): Final fairness metrics after transformation and training
- `validation_result` (ValidationResult): Validation result
- `model` (Any): Trained model object
- `transformed_df` (pd.DataFrame): Transformed DataFrame
- `predictions` (np.ndarray): Model predictions on test set
- `y_test` (np.ndarray | None): Test set labels (if available)
- `artifacts` (Dict[str, Any]): Additional artifacts

### `ValidationResult`

Validation result from workflow execution.

**Location:** `fairness_pipeline_dev_toolkit.integration.ValidationResult`

**Attributes:**
- `passed` (bool): Whether validation passed
- `baseline_metric_value` (float): Baseline metric value
- `final_metric_value` (float): Final metric value
- `threshold` (float | None): Validation threshold
- `improvement` (float): Improvement (negative means reduction in unfairness)
- `message` (str): Validation message

### `log_workflow_results()`

Log workflow results to MLflow.

**Location:** `fairness_pipeline_dev_toolkit.integration.log_workflow_results`

```python
def log_workflow_results(
    result: WorkflowResult,
    experiment_name: str,
    run_name: Optional[str] = None
) -> None
```

**Parameters:**
- `result` (WorkflowResult): Workflow execution result
- `experiment_name` (str): MLflow experiment name
- `run_name` (str, optional): MLflow run name

**Example:**
```python
from fairness_pipeline_dev_toolkit.integration import log_workflow_results

log_workflow_results(
    result=result,
    experiment_name="fairness_workflow",
    run_name="run_001"
)
```

### `to_markdown_report()`

Generate a markdown report from workflow results.

**Location:** `fairness_pipeline_dev_toolkit.integration.to_markdown_report`

```python
def to_markdown_report(
    result: WorkflowResult,
    output_path: str | Path
) -> None
```

**Parameters:**
- `result` (WorkflowResult): Workflow execution result
- `output_path` (str | Path): Path to save markdown report

**Example:**
```python
from fairness_pipeline_dev_toolkit.integration import to_markdown_report

to_markdown_report(result, "artifacts/report.md")
```

### `assert_fairness()`

Pytest plugin for asserting fairness in tests.

**Location:** `fairness_pipeline_dev_toolkit.integration.assert_fairness`

```python
def assert_fairness(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric: str = "demographic_parity_difference",
    threshold: float = 0.05,
    min_group_size: int = 30
) -> None
```

**Parameters:**
- `y_pred` (np.ndarray): Predictions
- `sensitive` (np.ndarray): Sensitive attribute values
- `metric` (str): Metric name (default: "demographic_parity_difference")
- `threshold` (float): Maximum allowed metric value (default: 0.05)
- `min_group_size` (int): Minimum group size (default: 30)

**Raises:** `AssertionError` if fairness threshold is exceeded

**Example:**
```python
import pytest
from fairness_pipeline_dev_toolkit.integration import assert_fairness

def test_model_fairness():
    y_pred = model.predict(X_test)
    assert_fairness(
        y_pred=y_pred,
        sensitive=gender_test,
        metric="demographic_parity_difference",
        threshold=0.05
    )
```

---

## Training

### `ReductionsWrapper`

Fairlearn reductions wrapper for scikit-learn models.

**Location:** `fairness_pipeline_dev_toolkit.training.ReductionsWrapper`

```python
from fairness_pipeline_dev_toolkit.training import ReductionsWrapper
from sklearn.linear_model import LogisticRegression

model = ReductionsWrapper(
    LogisticRegression(),
    constraint="demographic_parity",
    eps=0.01
)
model.fit(X_train, y_train, sensitive_features=A_train)
predictions = model.predict(X_test)
```

### `FairnessRegularizerLoss`

PyTorch loss function with fairness regularizer.

**Location:** `fairness_pipeline_dev_toolkit.training.FairnessRegularizerLoss`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.training import FairnessRegularizerLoss

criterion = FairnessRegularizerLoss(
    base_loss=nn.BCELoss(),
    eta=0.5,
    sensitive_attribute=sensitive
)
loss = criterion(predictions, targets)
```

### `LagrangianFairnessTrainer`

Lagrangian constraint-based trainer for PyTorch models.

**Location:** `fairness_pipeline_dev_toolkit.training.LagrangianFairnessTrainer`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.training import LagrangianFairnessTrainer

trainer = LagrangianFairnessTrainer(
    model=model,
    fairness="demographic_parity",
    dp_tol=0.02
)
trainer.train(X_train, y_train, sensitive_train)
```

### `GroupFairnessCalibrator`

Group-specific calibration for prediction scores.

**Location:** `fairness_pipeline_dev_toolkit.training.GroupFairnessCalibrator`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.training import GroupFairnessCalibrator

calibrator = GroupFairnessCalibrator(method="platt", min_samples=20)
calibrated_scores = calibrator.fit_transform(scores, y_true, groups)
```

### `sweep_pareto()` and `plot_pareto()`

Pareto frontier utilities for fairness-accuracy trade-offs.

**Location:** `fairness_pipeline_dev_toolkit.training.sweep_pareto`, `plot_pareto`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.training import sweep_pareto, plot_pareto

pareto_points = sweep_pareto(
    model_fn=lambda eta: train_model(eta=eta),
    etas=[0.0, 0.2, 0.5, 1.0]
)
plot_pareto(pareto_points, output_path="pareto.png")
```

---

## Monitoring

### `RealTimeFairnessTracker`

Real-time fairness metric tracking with sliding windows.

**Location:** `fairness_pipeline_dev_toolkit.monitoring.RealTimeFairnessTracker`

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    RealTimeFairnessTracker,
    TrackerConfig,
    ColumnMap
)

tracker = RealTimeFairnessTracker(
    TrackerConfig(window_size=10_000, min_group_size=30),
    artifacts_dir="artifacts/monitoring"
)

column_map = ColumnMap(
    y_true="y_true",
    y_pred="y_pred",
    sensitive="gender"
)

tracker.process_batch(df, column_map)
```

### `FairnessDriftAndAlertEngine`

Drift detection and alerting for production monitoring.

**Location:** `fairness_pipeline_dev_toolkit.monitoring.FairnessDriftAndAlertEngine`

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    FairnessDriftAndAlertEngine,
    DriftConfig
)

engine = FairnessDriftAndAlertEngine(
    DriftConfig(ks_threshold=0.05, alert_on_drift=True)
)

alerts = engine.check_drift(reference_metrics, current_metrics)
```

### `FairnessReportingDashboard`

Dashboard for visualizing fairness metrics over time.

**Location:** `fairness_pipeline_dev_toolkit.monitoring.FairnessReportingDashboard`

```python
from fairness_pipeline_dev_toolkit.monitoring import (
    FairnessReportingDashboard,
    ReportConfig
)

dashboard = FairnessReportingDashboard(
    ReportConfig(metrics_dir="artifacts/monitoring")
)
dashboard.generate_report(output_path="artifacts/report.html")
```

### `FairnessABTestAnalyzer`

A/B testing utilities for fairness comparisons.

**Location:** `fairness_pipeline_dev_toolkit.monitoring.FairnessABTestAnalyzer`

**Usage:**
```python
from fairness_pipeline_dev_toolkit.monitoring import FairnessABTestAnalyzer

analyzer = FairnessABTestAnalyzer()
results = analyzer.compare(
    group_a_metrics=metrics_a,
    group_b_metrics=metrics_b
)
```

---

## Exceptions

### Exception Hierarchy

All exceptions inherit from `FairnessToolkitError` and provide structured error information with user-friendly messages.

**Location:** `fairness_pipeline_dev_toolkit.exceptions`

```python
# Base exception
FairnessToolkitError

# Specific exceptions
ConfigValidationError      # Configuration validation failures
MetricComputationError     # Metric computation failures
PipelineExecutionError    # Pipeline execution failures
TrainingError             # Training failures
DataValidationError       # Data validation failures
DependencyError           # Missing optional dependencies
```

### Exception Attributes

All exceptions support:
- `message`: Human-readable error message
- `context`: Dictionary with additional error details
- `suggestion`: Optional suggestion for fixing the error

### Exception Types

#### `FairnessToolkitError`

Base exception for all toolkit errors.

```python
FairnessToolkitError(
    message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None
)
```

#### `ConfigValidationError`

Raised when configuration validation fails.

```python
ConfigValidationError(
    message: str,
    *,
    field_name: Optional[str] = None,
    field_value: Any = None,
    config_path: Optional[str] = None,
    suggestion: Optional[str] = None
)
```

**Example:**
```python
from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError

try:
    config = load_config("config.yml")
except ConfigValidationError as e:
    print(f"Error: {e.message}")
    print(f"Field: {e.context.get('field')}")
    print(f"Suggestion: {e.suggestion}")
```

#### `MetricComputationError`

Raised when metric computation fails.

```python
MetricComputationError(
    message: str,
    *,
    metric_name: Optional[str] = None,
    min_group_size: Optional[int] = None,
    actual_group_sizes: Optional[Dict[str, int]] = None,
    suggestion: Optional[str] = None
)
```

**Example:**
```python
from fairness_pipeline_dev_toolkit.exceptions import MetricComputationError

try:
    result = analyzer.demographic_parity_difference(...)
except MetricComputationError as e:
    print(f"Error: {e.message}")
    print(f"Group sizes: {e.context.get('group_sizes')}")
    print(f"Suggestion: {e.suggestion}")
```

#### `PipelineExecutionError`

Raised when pipeline execution fails.

```python
PipelineExecutionError(
    message: str,
    *,
    step_name: Optional[str] = None,
    step_index: Optional[int] = None,
    transformer_name: Optional[str] = None,
    suggestion: Optional[str] = None
)
```

#### `TrainingError`

Raised when training fails.

```python
TrainingError(
    message: str,
    *,
    method: Optional[str] = None,
    training_params: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None
)
```

#### `DataValidationError`

Raised when data validation fails.

```python
DataValidationError(
    message: str,
    *,
    missing_columns: Optional[list] = None,
    invalid_columns: Optional[Dict[str, str]] = None,
    data_shape: Optional[tuple] = None,
    suggestion: Optional[str] = None
)
```

#### `DependencyError`

Raised when required optional dependencies are missing.

```python
DependencyError(
    message: str,
    *,
    dependency_name: Optional[str] = None,
    extra_name: Optional[str] = None,
    suggestion: Optional[str] = None
)
```

**Example:**
```python
from fairness_pipeline_dev_toolkit.exceptions import DependencyError

try:
    from fairness_pipeline_dev_toolkit.training import ReductionsWrapper
except DependencyError as e:
    print(f"Error: {e.message}")
    print(f"Missing: {e.context.get('dependency')}")
    print(f"Install: {e.suggestion}")
```

### Usage Examples

**Basic Exception Handling:**
```python
from fairness_pipeline_dev_toolkit.exceptions import (
    FairnessToolkitError,
    ConfigValidationError,
    MetricComputationError
)

try:
    config = load_config("config.yml")
    result = analyzer.demographic_parity_difference(...)
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
    print(f"Suggestion: {e.suggestion}")
except MetricComputationError as e:
    print(f"Computation error: {e}")
    print(f"Context: {e.context}")
except FairnessToolkitError as e:
    print(f"Toolkit error: {e}")
```

**Accessing Exception Details:**
```python
try:
    # Some operation
    pass
except ConfigValidationError as e:
    # Access structured information
    print(f"Message: {e.message}")
    print(f"Field: {e.context.get('field')}")
    print(f"Value: {e.context.get('value')}")
    print(f"Suggestion: {e.suggestion}")
```

---

## Statistical Utilities

### `bootstrap_ci()`

Compute bootstrap confidence intervals.

**Location:** `fairness_pipeline_dev_toolkit.stats.bootstrap.bootstrap_ci`

```python
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci

ci = bootstrap_ci(
    data=samples,
    stat_fn=np.mean,
    level=0.95,
    method="percentile",
    B=1000
)
```

### `beta_binomial_interval()`

Compute Bayesian confidence intervals for binomial proportions.

**Location:** `fairness_pipeline_dev_toolkit.stats.bayesian.beta_binomial_interval`

```python
from fairness_pipeline_dev_toolkit.stats.bayesian import beta_binomial_interval

ci = beta_binomial_interval(successes=50, trials=100, level=0.95)
```

### `risk_ratio()` and `cohens_d()`

Effect size computations.

**Location:** `fairness_pipeline_dev_toolkit.stats.effect_size`

```python
from fairness_pipeline_dev_toolkit.stats.effect_size import risk_ratio, cohens_d

rr = risk_ratio(p1=0.6, p2=0.4)
d = cohens_d(group1_errors, group2_errors)
```

---

## Version Information

Get the toolkit version:

```python
from fairness_pipeline_dev_toolkit import __version__
print(__version__)  # "0.6.5"

# Alternatively, via the fairpipe namespace:
import fairpipe
print(fairpipe.__version__)  # "0.6.5"
```

---

## REST API

The REST API is an optional extra that exposes fairpipe over HTTP. It is intended for non-Python ML stacks and interactive demos via Swagger UI.

**Installation:**
```bash
pip install fairpipe[api]
```

**Start the server:**
```bash
fairpipe serve --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
# ReDoc:      http://localhost:8000/redoc
```

**Docker:**
```bash
docker build -t fairpipe-api .
docker run -p 8000:8000 fairpipe-api
# or: docker compose up
```

---

### `create_app()`

**Location:** `fairness_pipeline_dev_toolkit.api.app.create_app`

FastAPI application factory. Creates the app, attaches a `ResultStore` singleton to `app.state.store`, registers all routers, and installs the global exception handler.

```python
from fairness_pipeline_dev_toolkit.api.app import create_app

app = create_app()
```

---

### `ResultStore`

**Location:** `fairness_pipeline_dev_toolkit.api.store.ResultStore`

Thread-safe in-memory result store. Backed by `collections.OrderedDict` with LRU eviction when the cap is reached.

```python
ResultStore(maxsize: int = 500)
```

**Methods:**
- `put(run_id: str, result: dict) -> None` — store a result (evicts oldest if at capacity)
- `get(run_id: str) -> dict | None` — retrieve a result (returns `None` if not found)

---

### Endpoints

#### `GET /health`

Returns server version and current UTC timestamp.

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.7.1",
  "timestamp": "2026-05-07T10:00:00.000000+00:00"
}
```

---

#### `POST /validate`

Compute fairness metrics from JSON arrays. Results are stored in `ResultStore` and retrievable via `GET /results/{run_id}`.

**Request body:**
```json
{
  "y_pred":        [1, 0, 1, 0],
  "sensitive":     ["M", "F", "M", "F"],
  "y_true":        [1, 0, 0, 1],
  "y_score":       null,
  "with_ci":       false,
  "ci_level":      0.95,
  "with_effects":  false,
  "min_group_size": 5,
  "backend":       "native",
  "threshold":     0.05
}
```

**Required fields:** `y_pred`, `sensitive`

**Validation:** `len(y_pred)` must equal `len(sensitive)`. Returns `422` if lengths differ.

**Response 200:**
```json
{
  "run_id":  "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status":  "success",
  "passed":  true,
  "metrics": {
    "demographic_parity_difference": {
      "metric": "demographic_parity_difference",
      "value":  0.0312,
      "ci":     [0.0201, 0.0441],
      "effect_size": null,
      "n_per_group": {"M": 2, "F": 2}
    }
  },
  "timestamp": "2026-05-07T10:00:00.000000+00:00"
}
```

**Note:** `passed=false` (DPD > threshold) returns HTTP 200, not 500.

---

#### `POST /pipeline`

Run bias detection and mitigation on an uploaded CSV or Parquet file.

**Request:** `multipart/form-data`
- `file`: CSV or Parquet file upload
- `config`: YAML pipeline config string

**Response 200:**
```json
{
  "run_id": "...",
  "status": "success",
  "detector_report": { "meta": {}, "body": {} },
  "transformed_rows": 1000,
  "transformers_applied": ["reweigh"],
  "timestamp": "..."
}
```

**Error:** Returns `422` if the config has no `pipeline:` section.

---

#### `POST /workflow`

Execute the full 3-step workflow (baseline → transform+train → validate) on an uploaded file.

**Request:** `multipart/form-data`
- `file`: CSV or Parquet file upload
- `config`: YAML integrated workflow config string
- `min_group_size`: integer (optional, default `30`)
- `train_size`: float (optional, default `0.8`)

**Response 200:**
```json
{
  "run_id": "...",
  "status": "success",
  "validation": {
    "passed": true,
    "message": "Fairness threshold met.",
    "improvement": -0.312,
    "baseline_metric_value": 0.0814,
    "final_metric_value": 0.0312,
    "threshold": 0.05
  },
  "baseline_metrics": { "demographic_parity_difference": { "value": 0.0814, ... } },
  "final_metrics":    { "demographic_parity_difference": { "value": 0.0312, ... } },
  "timestamp": "..."
}
```

---

#### `GET /results/{run_id}`

Retrieve a stored result from any previous `/validate`, `/pipeline`, or `/workflow` call.

**Response 200:**
```json
{
  "run_id":     "3fa85f64-...",
  "endpoint":   "/validate",
  "result":     { ... },
  "created_at": "2026-05-07T10:00:00.000000+00:00"
}
```

**Response 404:**
```json
{
  "error":   "NotFound",
  "message": "No result found for run_id: 3fa85f64-..."
}
```

---

### Global Error Handler

Unhandled exceptions return HTTP 500:

```json
{
  "error":   "InternalServerError",
  "message": "<exception message>",
  "run_id":  null
}
```

`HTTPException` (e.g. `422` validation errors) passes through normally and is not affected by this handler.

---

## Backward Compatibility

The toolkit follows semantic versioning. Public APIs (classes and functions listed in this document) are stable within the same major version. Internal modules may change without notice.

For detailed information on versioning strategy, backward compatibility guarantees, deprecation policy, and migration guides, see the [Versioning Strategy](VERSIONING.md) document.

For questions or issues, see the [Integration Guide](integration_guide.md) or visit the [GitHub repository](https://github.com/SvrusIO/fAIr).
