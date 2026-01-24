# Changelog

All notable changes to the Fairness Pipeline Development Toolkit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.5.4] — 2026-01-24

### Added
- **Enhanced Testing Infrastructure**: Comprehensive testing improvements:
  - Added property-based testing with Hypothesis (`tests/property_based/test_property_based.py`)
  - Property-based tests verify invariants for bootstrap CI, effect sizes, and fairness metrics
  - Expanded integration tests (`tests/integration/test_integration_expanded.py`) with 20+ edge case scenarios
  - Edge case coverage includes: NaN/infinity handling, empty data, very large/small datasets, unicode characters, malformed data
  - Added `hypothesis>=6.100` to development dependencies

- **Documentation Site**: Complete documentation site infrastructure:
  - Created Sphinx documentation structure (`docs/conf.py`, `docs/index.rst`)
  - Set up automated documentation builds (`.github/workflows/docs.yml`)
  - Configured GitHub Pages deployment for automatic documentation hosting
  - Added ReadTheDocs configuration (`readthedocs.yml`) as alternative hosting option
  - Created getting started guide (`docs/getting_started.md`)
  - Documentation build automation with Makefile and requirements

- **Security Automation**: Ongoing security monitoring and automation:
  - Configured Dependabot (`.github/dependabot.yml`) for automated dependency updates
  - Weekly automated dependency security updates for GitHub Actions and pip packages
  - Security update grouping for efficient review
  - Created security review process documentation (`.github/SECURITY_REVIEW_PROCESS.md`)
  - Monthly automated security review workflow (`.github/workflows/security-review.yml`)
  - Security review includes: dependency scanning, code analysis (Bandit), Safety checks, Dependabot status

- **Release Workflow Improvements**: Enhanced automated release process:
  - Added TestPyPI support in release workflow (`.github/workflows/release.yml`)
  - Configurable repository selection (testpypi/pypi) via workflow inputs
  - Improved error handling: GitHub release creation no longer depends on PyPI publish success
  - Added `continue-on-error` to PyPI publish step to ensure releases are created even if publishing fails
  - Enhanced release notes with PyPI publication status and repository information
  - Support for both tag-based and manual workflow dispatch triggers

### Changed
- **Test Coverage**: Expanded test suite from 654 to 673 tests (86% coverage maintained)
- **Documentation**: Enhanced documentation with automated build and deployment workflows

### Improved
- **Code Quality**: All new code formatted with Black and passes linting checks
- **Testing**: Comprehensive edge case coverage for integration workflows
- **Security**: Automated security monitoring and review processes
- **Release Process**: More robust release workflow that ensures GitHub releases are created even if PyPI publishing encounters issues

### Purpose
This release focuses on testing infrastructure, documentation automation, and security automation. The enhanced testing ensures robustness across edge cases, the documentation site provides better user experience, and security automation ensures ongoing dependency security.

**Migration Notes**:
- No breaking changes to public APIs
- Property-based tests require `hypothesis` package (included in dev dependencies)
- Documentation site requires Sphinx and related packages (see `docs/requirements.txt`)
- Dependabot will automatically create pull requests for dependency updates
- Release workflow now supports TestPyPI by default; configure `TESTPYPI_API_TOKEN` secret in GitHub for TestPyPI publishing

---

## [v0.5.3] — 2026-01-24

### Added
- **Security Infrastructure**: Comprehensive security improvements:
  - Updated medium-priority dependencies with security fixes:
    - `fonttools>=4.60.2` (CVE-2025-66034)
    - `starlette>=0.49.1` (CVE-2025-62727)
    - `werkzeug>=3.1.5` (CVE-2025-66221, CVE-2026-21860)
    - `virtualenv>=20.36.2` (CVE-2026-22702)
  - Created automated security scanning workflow (`.github/workflows/security.yml`) with weekly scheduled scans
  - Added comprehensive security policy document (`SECURITY.md`) with vulnerability reporting guidelines
  - Updated `SECURITY_SCAN_RESULTS.md` to track remediation status

- **Performance Test Suite**: New performance testing infrastructure:
  - Created pytest-based performance test suite (`tests/performance/test_performance_suite.py`) with baseline performance tests
  - Added performance profiling script (`scripts/profile_performance.py`) using cProfile for bottleneck identification
  - Performance tests cover metrics computation, bootstrap CI, pipeline operations, and scalability
  - Tests establish performance baselines and detect regressions in CI/CD

- **Structured Logging**: Comprehensive logging infrastructure:
  - Implemented structured logging module (`fairness_pipeline_dev_toolkit/utils/logging.py`) with JSON format support
  - Added performance logging context manager for timing operations
  - Integrated logging into orchestrator, CLI, and monitoring modules
  - Supports log levels, contextual information (workflow IDs, step names), and performance timing
  - Configurable via environment variables (`FAIRPIPE_LOG_LEVEL`, `FAIRPIPE_LOG_FILE`, `FAIRPIPE_JSON_LOGS`)

- **User Feedback Collection**: Complete feedback infrastructure:
  - Created GitHub issue templates for bug reports, feature requests, and general feedback
  - Added comprehensive feedback documentation (`docs/FEEDBACK.md`) with feedback form and guidelines
  - Established feedback review process (`.github/FEEDBACK_REVIEW_PROCESS.md`) with priority guidelines and response timeframes
  - Configured issue template system with links to discussions and documentation

### Changed
- **Dependency Management**: Updated `pyproject.toml` and `requirements.in` to pin security-critical indirect dependencies
- **Performance Documentation**: Enhanced `docs/PERFORMANCE.md` with information about new performance test suite and profiling tools
- **Logging Integration**: All modules now use structured logging for better observability and debugging

### Improved
- **Code Quality**: Fixed linting and formatting issues across all new code
- **Test Coverage**: Added performance tests to test suite (654 total tests, 86% coverage)
- **Developer Experience**: Improved debugging capabilities with structured logging and performance profiling tools

### Purpose
This release focuses on production readiness improvements including security hardening, performance monitoring, observability through structured logging, and community engagement through feedback infrastructure. These enhancements support long-term maintainability and user satisfaction.

**Migration Notes**:
- No breaking changes to public APIs
- New logging is opt-in via environment variables (defaults to INFO level, console output)
- Security dependency updates are backward compatible
- Performance tests can be run independently: `pytest tests/performance/`

---

## [v0.5.2] — 2026-01-24

### Added
- **Performance Documentation**: Created comprehensive `docs/PERFORMANCE.md` with performance benchmarks, optimization tips, scalability considerations, memory usage guidelines, and CI/CD integration examples.

- **Automated Release Workflow**: Added `.github/workflows/release.yml` for automated PyPI publication and GitHub release creation when version tags are pushed.

- **Performance Benchmarking in CI**: Enhanced CI workflow to run performance benchmarks on Ubuntu, tracking performance regressions and uploading benchmark results as artifacts.

### Changed
- **Exception System**: Completely overhauled exception hierarchy with structured error types providing context and actionable suggestions:
  - Enhanced `FairnessToolkitError` base class with `message`, `context`, and `suggestion` attributes
  - Added `DataValidationError` for data validation failures
  - Added `DependencyError` for missing optional dependencies
  - All exceptions now provide user-friendly messages with installation suggestions

- **Error Messages**: Improved user-facing error messages throughout the codebase:
  - Configuration errors now include field names and suggestions
  - Training errors include method-specific installation instructions
  - Dependency errors provide exact pip install commands
  - Data validation errors list missing columns and data shapes

### Fixed
- **Optional Dependency Imports**: Fixed critical issue where core package required optional dependencies (torch, fairlearn) to be installed:
  - Made all training module imports lazy/conditional in `orchestrator.py`
  - Training classes (`ReductionsWrapper`, `LagrangianFairnessTrainer`, `FairnessRegularizerLoss`) are now only imported when needed
  - Core package can now be imported without `[training]` or `[adapters]` extras
  - Improved error messages when optional dependencies are missing

- **Orchestrator Module-Level Imports**: Removed top-level training imports from orchestrator, preventing import errors when optional dependencies are not installed.

### Improved
- **CI/CD Integration**: Enhanced integration guide with comprehensive CI/CD examples:
  - Performance benchmarking in CI/CD pipelines
  - Automated release workflow examples
  - GitHub Actions integration patterns

- **API Documentation**: Updated `docs/api.md` with complete exception hierarchy documentation, including all new exception types with usage examples.

- **Test Suite**: Updated orchestrator tests to use new `TrainingError` exception type instead of generic `ValueError`.

### Purpose
This release significantly improves the developer experience by fixing the optional dependency import issue, enhancing error messages with actionable suggestions, and adding comprehensive performance documentation. The automated release workflow streamlines the release process, and performance benchmarking in CI helps prevent performance regressions.

**Migration Notes**:
- No breaking changes to public APIs
- Core package can now be imported without optional dependencies
- Exception types are backward compatible (all inherit from `FairnessToolkitError`)
- Error messages are more informative but maintain same exception types

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

- **v0.5.4**: Enhanced testing infrastructure, documentation site, security automation, TestPyPI release workflow support
- **v0.5.3**: Security infrastructure, performance test suite, structured logging, user feedback collection
- **v0.5.2**: Optional dependency import fixes, enhanced error handling, performance documentation, automated release workflow
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

### v0.5.3
- **Dependency Updates**: Updated medium-priority dependencies with security fixes (fonttools, starlette, werkzeug, virtualenv)
- **Security Workflow**: Added automated weekly security scanning via GitHub Actions
- **Security Policy**: Established comprehensive security policy with vulnerability reporting guidelines

---

**Note**: Dates marked as "2025-01-XX" are placeholders. Update with actual release dates when known.
