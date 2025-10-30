# Changelog
<!-- A simple changelog for your first release candidate. Helps clients and teammates track what’s in v0.1.0. -->

## [v0.1.0] — 2025-10-30

### 🎯 Summary
First full release candidate of the **Fairness Pipeline Development Toolkit — Measurement Module**.  
Implements end-to-end fairness measurement, validation, and workflow integration.

---

### 🧩 Added
- **Metrics Engine**
  - Implemented `demographic_parity_difference`, `equalized_odds_difference`, and `mae_parity_difference`.
  - Supports intersectional groups with `min_group_size` filtering.
  - Unified `FairnessAnalyzer` wrapper for multiple libraries (Fairlearn, Aequitas, Native).

- **Statistical Validation**
  - Bootstrap confidence intervals (percentile, BCa).
  - Bayesian credible intervals for small-n.
  - Effect sizes (`risk_ratio`, `cohen_d`) to quantify disparity magnitude.

- **Workflow Integration**
  - `MLflow` logging of fairness metrics and structured artifacts.
  - `pytest` plugin with `assert_fairness()` for automated CI checks.
  - Command-line interface (`fairpipe validate`) for CSV-based validation.
  - Markdown reporting utility (`to_markdown_report()`).

- **Infrastructure**
  - Multi-environment CI (Ubuntu, macOS, Windows) for Python 3.10–3.12.
  - Linting (Ruff, Black, isort) and coverage reporting integrated into GitHub Actions.
  - Pre-commit hooks for code quality and optional fairness smoke checks.

- **Testing & Documentation**
  - End-to-end system tests validating full pipeline (`run → log → CI → report`).
  - Synthetic fixtures and coverage benchmarks.
  - `demo.ipynb` demonstrating metric computation, confidence intervals, and MLflow tracking.
  - Updated `README.md` and architecture record (`ADR-001-architecture.md`).

---

### 🧱 Structure

```markdown
fairness_pipeline_dev_toolkit/
├── measurement/ # Unified API entrypoint
├── metrics/ # Core metrics & adapters
├── stats/ # Bootstrap, Bayesian, effect sizes
├── integration/ # MLflow, pytest, reporting
├── cli/ # CLI (fairpipe)
└── utils/ # Validation & intersectional tools
```

### ✅ Status
**Release Candidate:** v0.1.0  
Validated via system tests, demo, and CI/CD pipeline.  
Ready for client pilot deployment and UX feedback collection.

