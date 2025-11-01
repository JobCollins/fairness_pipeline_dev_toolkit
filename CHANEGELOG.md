# Changelog

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
