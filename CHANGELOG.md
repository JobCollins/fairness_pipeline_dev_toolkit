# Changelog

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
