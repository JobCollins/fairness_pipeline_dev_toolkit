# ADR-001: Measurement Module Architecture

**Context:**  
The Measurement Module unifies fairness measurement across ML workflows, providing a defensible, reproducible, and extensible layer for evaluating fairness metrics.  
It standardizes how statistical validation and reporting are handled across multiple teams, libraries, and pipelines.

---

### **Decision**

1. **Single Unified Entry Point**  
   - All fairness computations are exposed through the `FairnessAnalyzer` class, located in the `measurement` module.  
   - The analyzer supports multiple backends via adapters:  
     - `native` (built-in reference implementation)  
     - `FairlearnAdapter`  
     - `AequitasAdapter`  
   - This abstraction allows seamless switching between fairness libraries without changing user code.

2. **Metrics Engine**
   - Implements parity metrics for both **classification** and **regression** tasks:  
     - `demographic_parity_difference`  
     - `equalized_odds_difference`  
     - `mae_parity_difference`  
   - Supports **intersectional analysis** with `min_group_size` filtering to ensure statistical reliability.

3. **Statistical Validation Layer**
   - Provides bootstrap-based confidence intervals (`bootstrap_ci`, `bca_ci`) and Bayesian small-n corrections (`beta_binomial_interval`).  
   - Reports standardized effect sizes (`risk_ratio`, `cohen_d`) for practical interpretability.  
   - Each metric returns a structured result object:  
     ```python
     {
       "metric": "equalized_odds_difference",
       "value": 0.12,
       "ci": [0.09, 0.17],
       "effect_size": 1.42,
       "n_per_group": {"A": 120, "B": 100}
     }
     ```

4. **Workflow Integration**
   - **MLflow logger:** Records fairness metrics, confidence intervals, and metadata as structured artifacts.  
   - **Pytest plugin:** Adds `assert_fairness()` for automated threshold checks within CI/CD pipelines.  
   - **CLI tool (`fairpipe validate`):** Enables one-command validation using a CSV input.  
   - **Markdown reporting:** Generates readable fairness summaries for audits and documentation.

5. **Development Workflow**
   - Pre-commit hooks ensure linting (Black, Ruff, isort) and smoke fairness checks.  
   - CI (GitHub Actions) runs matrix tests across Python 3.10–3.12 and multiple OS environments.  
   - System tests confirm end-to-end flow: `metric → CI → MLflow → report`.

---

### **Consequences**
- Developers now use a **single, stable interface** instead of juggling multiple fairness libraries.  
- The measurement process is **statistically robust**, traceable, and CI-ready.  
- The modular architecture supports future extensions (e.g., group fairness dashboards, temporal bias tracking).

---

### **Status:**  
✅ **Accepted (v0.1.0)** — Measurement Module released and validated through demo and CI integration.

---

# ADR-002: `fairpipe` Shim Namespace

**Date:** 2026-05-07

---

### **Context**

The project is published to PyPI under the name `fairpipe`, but the internal Python package was
named `fairness_pipeline_dev_toolkit`. This mismatch meant that `pip install fairpipe` succeeded
but `import fairpipe` raised `ModuleNotFoundError`, forcing users to discover and use the internal
package name. This violated the principle of least surprise and created friction in onboarding,
documentation, and third-party integrations.

---

### **Decision**

A `fairpipe/` top-level shim package was added to the repository. Each submodule is a thin
re-export that imports all public symbols from the corresponding `fairness_pipeline_dev_toolkit`
submodule using `from fairness_pipeline_dev_toolkit.X import *` and forwards `__all__`. Object
identity is fully preserved — no wrappers, no copies.

The shim covers all public submodules: `metrics`, `pipeline`, `training`, `monitoring`,
`integration`, and `exceptions`. The package-level `__version__` is re-exported from
`fairness_pipeline_dev_toolkit.__version__`.

---

### **Consequences**

- Users can now write `import fairpipe` or `from fairpipe.metrics import FairnessAnalyzer` as
  naturally as expected from the PyPI package name.
- All existing code using `fairness_pipeline_dev_toolkit.*` continues to work without any changes
  (full backward compatibility).
- Both namespaces are documented as equally valid and maintained in lockstep.
- The shim adds no runtime overhead beyond a single extra import indirection per submodule on
  first import.

---

### **Status:**
✅ **Accepted (v0.6.5)** — `fairpipe` shim namespace shipped, tested (4 namespace parity tests), and documented.
