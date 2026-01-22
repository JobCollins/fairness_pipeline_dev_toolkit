# Changelog

All notable changes to the Fairness Pipeline Development Toolkit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.5.1] — 2025-01-16

### Fixed
- **Critical Bug Fix**: Fixed `IndexError` when using categorical Series with NaN values in intersectional analysis. The `_intersectional_prep()` function now properly handles categorical Series conversion by converting to string first, then to numpy array with proper NaN handling. This resolves issues in `demographic_parity_difference()`, `equalized_odds_difference()`, and `mae_parity_difference()` methods when using intersectional analysis with categorical data.

- **CLI Test Reliability**: Fixed CLI tests to properly handle `SystemExit` exceptions from argparse, improving test reliability.

- **AB Test Assertions**: Updated AB test assertions to account for bootstrap sampling variability, reducing false test failures.

- **Edge Case Handling**: Fixed intersectional tests to handle empty DataFrame edge cases and resolved variable name conflicts in CLI command tests.

### Improved
- **Test Suite Quality**: Comprehensive test suite overhaul with all 645 tests now passing:
  - Tests updated to create sample files within test functions instead of relying on external files (improves isolation and portability)
  - Enhanced assertions in smoke tests to validate actual outputs
  - Added negative test cases for error handling in CLI commands and config loading
  - Improved test organization with better separation of concerns
  - Expanded test coverage from 68% to 87% across all modules

- **Test Documentation**: Enhanced `TEST_REVIEW_REPORT.md` with comprehensive test suite significance analysis, including detailed evaluation of 9 major test suites with star ratings and insights on test suite prioritization.

### Testing
- **Test Coverage**: All 645 tests passing across all modules with 87% code coverage:
  - Core/Measurement: 7 test files
  - Pipeline: 7 test files (including new unit tests for transformers and detectors)
  - Integration: 3 test files
  - System/E2E: 3 test files
  - CLI: 2 test files
  - Training: 5 test files
  - Monitoring: 3 test files (including new AB test coverage)
  - Utils: 2 test files (new coverage for intersectional and validation)
  - Stats: 3 test files (new coverage for effect_size, multipletests, bayesian)
  
- **Coverage Improvements**: Test coverage increased from 68% to 87%, with comprehensive coverage across:
  - Metrics computation and adapters
  - Pipeline transformers and detectors
  - Integration workflows and orchestrator
  - Training methods (reductions, regularized, lagrangian)
  - Monitoring tools and drift detection
  - Statistical validation functions

### Purpose
This patch release addresses a critical bug in intersectional analysis that could cause failures with categorical data, and significantly improves test suite reliability and documentation. The test suite now provides comprehensive coverage with all tests passing, ensuring the toolkit's reliability and correctness.

---

## [v0.5.0] — 2025-01-XX

### Added
- **Integrated End-to-End Workflow**: Introduced unified three-step workflow orchestrator combining Measurement, Pipeline, and Training modules into a single automated process.

- **New CLI Command**: `fairpipe run-pipeline` executes complete workflow:
  1. Baseline Measurement - audit raw data for fairness issues
  2. Transform Data + Train Model - apply bias mitigation and train fairness-aware model
  3. Final Validation - compare metrics to baseline and validate against threshold

- **Extended Config Schema**: Added `training`, `fairness_metric`, and `validation_threshold` fields to support integrated workflow. Config files can now specify training method (reductions, regularized, lagrangian) with method-specific parameters.

- **Complete MLflow Integration**: Enhanced MLflow logger to log complete workflow results including baseline/final metrics, validation status, model artifacts, and config.yml. Enables tracking of fairness experiments over time.

- **Integrated Demo Notebook**: Created `demo_integrated.ipynb` demonstrating the complete integrated workflow from raw data to validated model.

### Changed
- **Config System**: Extended `PipelineConfig` to support training method selection with method-specific parameters. Configs without a `training` section continue to work for pipeline-only execution via `fairpipe pipeline` command.

- **Orchestrator**: Enhanced to handle sensitive attribute encoding for PyTorch models and proper feature matrix construction. Sensitive attributes are now automatically excluded from feature matrices before training.

- **Result Object Handling**: Validation function now correctly handles both Result objects and dict formats, improving compatibility.

### Fixed
- **Feature Matrix Construction**: Sensitive attributes are now properly excluded from feature matrices before training, preventing data leakage.

- **Sensitive Attribute Encoding**: String sensitive attributes are automatically encoded as integers for PyTorch models, preventing type errors.

- **Result Object Handling**: Validation function now correctly handles both Result objects and dict formats.

### Testing
- **Test Coverage**: Added comprehensive integration test suite (22 new tests):
  - Config schema validation with training section (8 tests)
  - Orchestrator workflow functions (9 tests)
  - MLflow workflow logging (5 tests)
  - CLI end-to-end integration (3 tests)

### Purpose
This major update transforms the toolkit from modular components into a unified, integrated system. Users can now execute a complete fairness workflow from raw data to validated model with a single command, with automatic baseline comparison and threshold validation. This enables CI/CD integration and automated fairness assurance.

**Migration Notes**: 
- Configs without a `training` section continue to work with `fairpipe pipeline` command
- To use integrated workflow, add `training` section to config and use `fairpipe run-pipeline` command
- No breaking changes to existing CLI commands or APIs

---

## [v0.4.2] — 2025-01-XX

### Fixed
- **RealTimeFairnessTracker**: Fixed to use `DatetimeIndex` instead of timestamp column, ensuring proper time-series format as required. Metrics are now stored with timestamp as the index, improving time-series analysis capabilities.

- **FairnessDriftAndAlertEngine**: Enhanced alert severity scoring to incorporate group size (n) from metrics. Smaller groups now reduce confidence in drift detection, preventing false alarms from statistically unreliable samples.

- **FairnessReportingDashboard**: Updated to handle `DatetimeIndex` format with backward compatibility for timestamp column format.

### Changed
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
This update addresses critical gaps identified in the Monitoring Module assessment, ensuring proper time-series format with DatetimeIndex, complete alert prioritization logic including group size, comprehensive demo notebook demonstrating all components, and improved visualization with heatmap for intersectional analysis.

**Migration Notes**:
- Monitoring apps now expect `DatetimeIndex` format in metrics CSV files
- Old timestamp column format is still supported for backward compatibility
- No breaking changes to API interfaces

---

## [v0.4.1] — 2025-11-19

### Fixed
- **ReductionsWrapper**: Fixed `T` parameter not being passed to `ExponentiatedGradient`. The parameter is now correctly forwarded as `max_iter` to control iteration limits.

### Changed
- **Pareto Visualization**: Enhanced `sweep_pareto()` to automatically save plots when `save_path` is provided, streamlining the workflow for generating and saving Pareto frontier visualizations.

### Added
- **Training Module Demo**: Created `demo_training.ipynb` providing comprehensive examples demonstrating all Training Module components with synthetic data generation, visualizations, and usage patterns.

### Testing
- **Test Coverage**: Expanded test suite with comprehensive edge case testing:
  - ReductionsWrapper: T parameter verification, kwargs override, multiple constraint types
  - Pareto Visualization: save_path functionality, plot generation
  - FairnessRegularizerLoss: single group scenarios, eta edge cases, invalid mode handling
  - GroupFairnessCalibrator: small groups, missing groups, multiple groups, empty inputs

### Purpose
This update addresses critical gaps identified in the Training Module assessment, ensuring all components are properly documented, tested, and functional. The ReductionsWrapper fix ensures proper iteration control, and the expanded test coverage improves reliability.

**Migration Notes**:
- No breaking changes
- ReductionsWrapper now correctly respects `T` parameter for iteration limits

---

## [v0.4.0] — 2025-11-01

### Added
- **Training Module**: Introduced a new module enabling fairness-aware model training, bridging fair data pipelines with fair models. Added components:
  - **ReductionsWrapper (scikit-learn)**: Integrates `fairlearn.reductions.ExponentiatedGradient` for training under fairness constraints (e.g., Demographic Parity).
  - **FairnessRegularizer (PyTorch)**: Introduces fairness penalties directly into loss functions for differentiable fairness optimization.
  - **LagrangianFairnessTrainer (PyTorch)**: Performs constrained optimization via Lagrange multipliers to enforce Demographic Parity or Equal Opportunity.
  - **GroupFairnessCalibrator**: Post-training correction of prediction probabilities using Platt Scaling or Isotonic Regression.
  - **ParetoFrontier Visualization Tool**: Plots the fairness–accuracy trade-off across varying regularization strengths.

- **CLI Commands**: Added new CLI commands for training:
  - `fairpipe train-regularized`: Train NN with fairness regularizer and generate Pareto frontier
  - `fairpipe train-lagrangian`: Train NN with Lagrangian fairness constraints
  - `fairpipe calibrate`: Apply group-specific calibration to prediction scores

- **Optional Dependencies**: Added `[training]` extra for PyTorch and related dependencies.

### Changed
- **Unified CLI Configuration**: Refined CLI configuration and profile loading (`pipeline.config.yml`) to support both *pipeline* and *training* profiles.

- **Exception Handling**: Refined exception handling for `ExponentiatedGradient` compatibility and PyTorch gradient tracking.

### Testing
- **Test Coverage**: Expanded automated test coverage under `tests/training/` for sklearn, torch, postproc, and visualization submodules.

### Purpose
Phase 6 extends the toolkit's capabilities beyond data-level fairness by embedding fairness constraints directly into model training workflows, ensuring equitable outcomes by design.

**Migration Notes**:
- Training module requires `pip install -e .[training]` to enable PyTorch dependencies
- PyTorch installation may vary by platform (see PyTorch installation guide)
- No breaking changes to existing pipeline or measurement modules

---

## [v0.3.0-rc1] — 2025-10-31

### Added
- **System Test**: End-to-end CLI test (`tests/system/test_cli_e2e_pipeline.py`) verifying full pipeline execution and artifact generation.

- **Demo Notebook Generator**: `scripts/make_demo_notebook.py` programmatically creates a clean, runnable `demo.ipynb` showing detection → mitigation → reporting.

- **Artifacts**: Auto-generated `demo.ipynb` ready for Jupyter or VS Code use.

### Changed
- **Documentation**: Expanded README with Phase 5 instructions (E2E tests, demo generation, and MLflow logging).

- **Test Reliability**: Improved test reliability for pipeline and detector integration.

### Purpose
Phase 5 finalized the first release candidate by validating the entire fairness pipeline through automated tests and a reproducible demo.

---

## Version History Summary

- **v0.5.1**: Critical intersectional analysis bug fix, test suite improvements
- **v0.5.0**: Major release with integrated end-to-end workflow
- **v0.4.2**: Monitoring module improvements (DatetimeIndex, heatmaps, alert scoring)
- **v0.4.1**: Training module fixes and expanded test coverage
- **v0.4.0**: Training module introduction (reductions, regularizers, Lagrangian)
- **v0.3.0-rc1**: First release candidate with system tests and demo notebooks

---

## Breaking Changes

### v0.5.0
- **None**: All changes are backward compatible. Configs without `training` section continue to work with `fairpipe pipeline` command.

### v0.4.2
- **Monitoring Format**: Monitoring apps now expect `DatetimeIndex` format, but backward compatibility maintained for timestamp columns.

### v0.4.0
- **None**: Training module is additive, no breaking changes to existing modules.

---

## Deprecations

No deprecations in current version.

---

## Security

No security-related changes in documented versions.

---

**Note**: Dates marked as "2025-01-XX" are placeholders. Update with actual release dates when known.
