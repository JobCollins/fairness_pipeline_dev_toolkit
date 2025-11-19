# Changelog

## [v0.5.0] — 2025-01-XX
### Added
- **Integrated End-to-End Workflow**
  - Introduced unified three-step workflow orchestrator combining Measurement, Pipeline, and Training modules
  - **New CLI Command**: `fairpipe run-pipeline` executes complete workflow:
    1. Baseline Measurement - audit raw data for fairness issues
    2. Transform Data + Train Model - apply bias mitigation and train fairness-aware model
    3. Final Validation - compare metrics to baseline and validate against threshold
  - **Extended Config Schema**: Added `training`, `fairness_metric`, and `validation_threshold` fields to support integrated workflow
  - **Complete MLflow Integration**: Logs accuracy, fairness metrics, model artifacts, and config.yml to MLflow
  - **Integrated Demo Notebook**: Created `demo_integrated.ipynb` demonstrating the complete workflow

### Improved
- **Config System**: Extended `PipelineConfig` to support training method selection (reductions, regularized, lagrangian) with method-specific parameters
- **MLflow Logger**: Enhanced to log complete workflow results including baseline/final metrics, validation status, and all artifacts
- **Orchestrator**: Handles sensitive attribute encoding for PyTorch models and proper feature matrix construction
- **Documentation**: Added architecture diagram (Mermaid) and comprehensive integrated configuration guide to README

### Fixed
- **Feature Matrix Construction**: Sensitive attributes are now properly excluded from feature matrices before training
- **Result Object Handling**: Validation function now correctly handles both Result objects and dict formats
- **Sensitive Attribute Encoding**: String sensitive attributes are automatically encoded as integers for PyTorch models

### Testing
- **Test Coverage**: Comprehensive integration test suite (22 new tests):
  - Config schema validation with training section (8 tests)
  - Orchestrator workflow functions (9 tests)
  - MLflow workflow logging (5 tests)
  - CLI end-to-end integration (3 tests)

### Purpose
This major update transforms the toolkit from modular components into a unified, integrated system. Users can now execute a complete fairness workflow from raw data to validated model with a single command, with automatic baseline comparison and threshold validation.

---

## [v0.4.2] — 2025-01-XX
### Fixed
- **RealTimeFairnessTracker**: Fixed to use `DatetimeIndex` instead of timestamp column, ensuring proper time-series format as required. Metrics are now stored with timestamp as the index, improving time-series analysis capabilities.
- **FairnessDriftAndAlertEngine**: Enhanced alert severity scoring to incorporate group size (n) from metrics. Smaller groups now reduce confidence in drift detection, preventing false alarms from statistically unreliable samples.
- **FairnessReportingDashboard**: Updated to handle `DatetimeIndex` format with backward compatibility for timestamp column format.

### Improved
- **FairnessReportingDashboard**: Converted intersectional visualization from bar chart to heatmap (`go.Heatmap`) for better visualization of fairness metrics across intersectional subgroups. Heatmap uses diverging colormap (RdYlBu_r) to highlight disparities.
- **FairnessDriftAndAlertEngine**: Severity scoring now considers group size with confidence factors:
  - Groups with n < 30: Reduced confidence (penalized severity)
  - Groups with 30 ≤ n < 100: Gradual confidence increase
  - Groups with n ≥ 100: Full confidence
- **Monitoring Apps**: Updated Streamlit and Dash apps to properly load CSV files with `DatetimeIndex`, maintaining compatibility with new format.

### Added
- **Monitoring Module Demo**: Created `demo_monitoring.ipynb` that simulates a production stream and demonstrates all monitoring components working together:
  - RealTimeFairnessTracker processing batches over time
  - FairnessDriftAndAlertEngine detecting drift and generating alerts
  - FairnessReportingDashboard visualizing trends and intersectional metrics
  - FairnessABTestAnalyzer for A/B testing scenarios
- **Test Coverage**: Added comprehensive test suite for monitoring module:
  - `tests/monitoring/test_tracker.py`: Tests for tracker DatetimeIndex usage, CSV persistence, sliding window, and metric computation
  - Updated `tests/monitoring/test_dashboard_and_drift.py`: Tests for DatetimeIndex handling, heatmap visualization, and severity scoring with group size

### Purpose
This update addresses critical gaps identified in the Monitoring Module assessment, ensuring:
- Proper time-series format with DatetimeIndex
- Complete alert prioritization logic including group size
- Comprehensive demo notebook demonstrating all components
- Improved visualization with heatmap for intersectional analysis

---

## [v0.4.1] — 2025-11-19
### Fixed
- **ReductionsWrapper**: Fixed `T` parameter not being passed to `ExponentiatedGradient`. The parameter is now correctly forwarded as `max_iter` to control iteration limits.

### Improved
- **Pareto Visualization**: Enhanced `sweep_pareto()` to automatically save plots when `save_path` is provided, streamlining the workflow for generating and saving Pareto frontier visualizations.
- **Test Coverage**: Expanded test suite with comprehensive edge case testing:
  - ReductionsWrapper: T parameter verification, kwargs override, multiple constraint types
  - Pareto Visualization: save_path functionality, plot generation
  - FairnessRegularizerLoss: single group scenarios, eta edge cases, invalid mode handling
  - GroupFairnessCalibrator: small groups, missing groups, multiple groups, empty inputs

### Added
- **Training Module Demo**: Created `demo_training.ipynb` providing comprehensive examples demonstrating all Training Module components with synthetic data generation, visualizations, and usage patterns.

### Purpose
This update addresses critical gaps identified in the Training Module assessment, ensuring all components are properly documented, tested, and functional.

---

## [v0.4.0] — 2025-11-01
### Added
- **Training Module**
  - Introduced a new module enabling fairness-aware **model training**, bridging fair data pipelines with fair models.
  - Added components:
    - **ReductionsWrapper (scikit-learn)** — integrates `fairlearn.reductions.ExponentiatedGradient` for training under fairness constraints (e.g., Demographic Parity).
    - **FairnessRegularizer (PyTorch)** — introduces fairness penalties directly into loss functions for differentiable fairness optimization.
    - **LagrangianFairnessTrainer (PyTorch)** — performs constrained optimization via Lagrange multipliers to enforce Demographic Parity or Equal Opportunity.
    - **GroupFairnessCalibrator** — post-training correction of prediction probabilities using Platt Scaling or Isotonic Regression.
    - **ParetoFrontier Visualization Tool** — plots the fairness–accuracy trade-off across varying regularization strengths.
  - Fully compatible with Python **3.12.5** and macOS environments.

### Improved
- Unified CLI configuration and profile loading (`pipeline.config.yml`) to support both *pipeline* and *training* profiles.
- Refined exception handling for `ExponentiatedGradient` compatibility and PyTorch gradient tracking.
- Expanded automated test coverage under `tests/training/` for sklearn, torch, postproc, and visualization submodules.
- Streamlined documentation to include CLI commands and developer setup for the new module.

### Purpose
Phase 6 extends the toolkit’s capabilities beyond data-level fairness by embedding fairness constraints directly into **model training workflows**, ensuring equitable outcomes by design.

---

## [v0.3.0-rc1] — 2025-10-31
### Added
- **System Test:** End-to-end CLI test (`tests/system/test_cli_e2e_pipeline.py`) verifying full pipeline execution and artifact generation.
- **Demo Notebook Generator:** `scripts/make_demo_notebook.py` programmatically creates a clean, runnable `demo.ipynb` showing detection → mitigation → reporting.
- **Artifacts:** Auto-generated `demo.ipynb` ready for Jupyter or VS Code use.
- **Docs Update:** Expanded README with Phase 5 instructions (E2E tests, demo generation, and MLflow logging).

### Improved
- Documentation flow and developer onboarding clarity.
- Test reliability for pipeline and detector integration.

### Purpose
Phase 5 finalized the first release candidate by validating the entire fairness pipeline through automated tests and a reproducible demo.
