# API Reference

Complete API documentation for the Fairness Pipeline Development Toolkit.

---

## Table of Contents

- [Core Metrics](#core-metrics)
- [Pipeline Utilities](#pipeline-utilities)
- [Integration & Workflow](#integration--workflow)
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

**Methods:**

#### `demographic_parity_difference()`

Compute the demographic parity difference (DPD) metric.

```python
def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray | pd.Series,
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
- `y_pred` (np.ndarray): Binary predictions (0/1) or continuous scores
- `sensitive` (np.ndarray | pd.Series): Sensitive attribute values
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
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray | pd.Series,
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
- `y_true` (np.ndarray): Ground truth binary labels (0/1)
- `y_pred` (np.ndarray): Binary predictions (0/1)
- `sensitive` (np.ndarray | pd.Series): Sensitive attribute values
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray | pd.Series,
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
- `y_true` (np.ndarray): Ground truth continuous values
- `y_pred` (np.ndarray): Predicted continuous values
- `sensitive` (np.ndarray | pd.Series): Sensitive attribute values
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

All exceptions inherit from `FairnessToolkitError`:

**Location:** `fairness_pipeline_dev_toolkit.exceptions`

```python
# Base exception
FairnessToolkitError

# Specific exceptions
ConfigValidationError      # Configuration validation failures
MetricComputationError     # Metric computation failures
PipelineExecutionError    # Pipeline execution failures
TrainingError             # Training failures
```

**Usage:**
```python
from fairness_pipeline_dev_toolkit.exceptions import (
    FairnessToolkitError,
    ConfigValidationError,
    MetricComputationError
)

try:
    config = load_config("config.yml")
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
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

print(__version__)  # "0.5.0"
```

---

## Backward Compatibility

The toolkit follows semantic versioning. Public APIs (classes and functions listed in this document) are stable within the same major version. Internal modules may change without notice.

For detailed information on versioning strategy, backward compatibility guarantees, deprecation policy, and migration guides, see the [Versioning Strategy](VERSIONING.md) document.

For questions or issues, see the [Integration Guide](integration_guide.md) or visit the [GitHub repository](https://github.com/SvrusIO/fAIr).
