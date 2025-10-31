# Changelog
<!-- A simple changelog for your first release candidate. Helps clients and teammates track what’s in v0.1.0. -->
## [v0.2.0] — 2025-10-31
# Changelog

## [v0.2.0] — 2025-10-31
### Added
- **Pipeline Module (Phases 3 & 4)**
  - Implemented YAML-driven configuration (`pipeline.config.yml`) for sensitive attributes, benchmarks, and pipeline steps.
  - Integrated a **Bias Detection Engine** (representation, disparity, and proxy analysis) returning typed `BiasReport` objects.
  - Added orchestration engine for end-to-end execution (`run_detectors`, `build_pipeline`, `apply_pipeline`).
  - Introduced scikit-learn compatible transformers:
    - `InstanceReweighting`
    - `DisparateImpactRemover`
    - `ReweighingTransformer`
    - `ProxyDropper`
  - Expanded transformer registry for safer instantiation and automatic default injection.

### Changed
- Enhanced CLI command `pipeline` to support:
  - `--config`, `--csv`, `--out-csv`, `--detector-json`, and `--report-md` arguments.
  - Full execution of bias detection → transformation → artifact generation.
- Refactored engine logic for compatibility with `BiasReport.to_dict()` serialization.
- Improved validation for YAML configs and transformer parameters.

### Fixed
- Resolved serialization errors (`BiasReport` JSON conversion).
- Fixed import path consistency across installed and editable environments.
- Eliminated “Unknown transformer” and missing parameter errors by unifying registry defaults.

### Deliverables
- CLI workflow validated:
  ```bash
  python -m fairness_pipeline_dev_toolkit.cli.main pipeline \
    --config fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml \
    --csv dev_sample.csv \
    --out-csv artifacts/sample.transformed.csv \
    --detector-json artifacts/detectors.json \
    --report-md artifacts/pipeline_run.md
  ```

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
