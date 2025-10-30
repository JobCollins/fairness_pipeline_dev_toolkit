# Fairness Pipeline Development Toolkit — Measurement Module


A unified, statistically-rigorous **fairness measurement** layer for machine learning workflows.
The toolkit wraps multiple libraries (Fairlearn, Aequitas) behind a single API, computes bootstrap and Bayesian confidence intervals, reports standardized effect sizes, and integrates seamlessly with MLflow, pytest, and CI/CD pipelines.


## Features
- Unified `FairnessAnalyzer` with adapters (Fairlearn/Aequitas)
- Classification & regression fairness metrics, intersectional analysis
- Bootstrap (and small‑n Bayesian) intervals for robust statistical validation
- Effect size computation (risk ratio, Cohen’s d) for practical disparity magnitude
- **MLOps integration**: MLflow logging, pytest assertions, and CLI (`fairpipe validate`)
- Modular architecture with a user-friendly public API under `measurement/`


## Quick start: Installation

```console
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
pytest -q
```



## Measurement Module Overview

The Measurement Module is the public entry point of the toolkit.
It unifies all fairness-related functionality — metrics, statistical validation, reporting, and integration — under a single import path.

### Example Usage


```python
from fairness_pipeline_dev_toolkit.measurement import (
    FairnessAnalyzer,
    bootstrap_ci,
    beta_binomial_interval,
    risk_ratio,
    to_markdown_report,
)

# 1. Initialize the analyzer
fa = FairnessAnalyzer(min_group_size=30, backend="native")

# 2. Compute fairness metrics
res_dp = fa.demographic_parity_difference(y_pred, sensitive)
res_eq = fa.equalized_odds_difference(y_true, y_pred, sensitive)

# 3. Compute confidence intervals
ci = bootstrap_ci(y_pred, np.mean, B=1000, level=0.95)

# 4. Generate Markdown report
md = to_markdown_report({"dp": res_dp, "eo": res_eq})
print(md)
```

## Demo and Sytem Tests
Run the full end-to-end system tests:

```console
pytest tests/system -q
```

Generate a quick sample fairness report with CI and effect sizes:

```console
python -m fairness_pipeline_dev_toolkit.cli.main validate \
  --csv dev_sample.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive sensitive \
  --backend native \
  --min-group-size 2 \
  --with-ci --ci-level 0.95 --bootstrap-B 1000 \
  --with-effects \
  --out artifacts/report_with_ci.md
```

## Project Layout

```markdown
fairness_pipeline_dev_toolkit/
├── cli/              # CLI and command-line interface (fairpipe)
├── metrics/          # Adapters + core metric definitions
├── stats/            # Bootstrap, Bayesian, effect size utilities
├── integration/      # MLflow, pytest, and reporting integrations
├── utils/            # Validation, intersectional utilities
└── measurement/      # Unified public API (Measurement Module)
```

## Documentation

See `docs/ADR-001-architecture.md` for design decisions.
The API is stable as of version `v0.1.0`.













