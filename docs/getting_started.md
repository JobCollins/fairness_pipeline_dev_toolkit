# Getting Started

Welcome to the Fairness Pipeline Development Toolkit! This guide will help you get started quickly.

## Installation

Install the toolkit using pip:

```bash
pip install fairpipe
```

For development installation:

```bash
git clone https://github.com/SvrusIO/fAIr
cd fAIr
pip install -e .[dev]
```

## Quick Start

### Basic Usage

`FairnessAnalyzer` defaults to `min_group_size=30`. Groups smaller than that are
excluded, and the metric is undefined (`nan`) until each group clears the guard.
That is intentional — a disparity from four rows per group is not evidence.

```python
from fairpipe.metrics import FairnessAnalyzer
import numpy as np

fa = FairnessAnalyzer(min_group_size=30, backend="native")

# Four rows per group — below the default guard
y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
sensitive = np.array(["A", "A", "B", "B", "A", "B", "A", "B"])

result = fa.demographic_parity_difference(y_pred, sensitive)

print(f"Demographic Parity Difference: {result.value}")  # nan
print(f"95% CI: {result.ci}")  # None
print(f"n_per_group: {result.n_per_group}")  # empty / excluded
```

With enough rows per group, the same call returns a finite value:

```python
from fairpipe.metrics import FairnessAnalyzer
import numpy as np

rng = np.random.default_rng(0)
n = 40  # per group; clears default min_group_size=30
y_pred = np.concatenate([rng.integers(0, 2, n), rng.integers(0, 2, n)])
sensitive = np.array(["A"] * n + ["B"] * n)

fa = FairnessAnalyzer(min_group_size=30, backend="native")
result = fa.demographic_parity_difference(y_pred, sensitive, with_ci=True)

print(f"Demographic Parity Difference: {result.value:.4f}")
print(f"95% CI: {result.ci}")
print(f"n_per_group: {result.n_per_group}")
```

### DataFrame Proxy

If your data is already in a DataFrame, bind a proxy to avoid repeating column names:

```python
import pandas as pd
import numpy as np
from fairpipe.metrics import FairnessAnalyzer

rng = np.random.default_rng(0)
n = 40
df = pd.DataFrame({
    "y_pred": np.concatenate([rng.integers(0, 2, n), rng.integers(0, 2, n)]),
    "y_true": np.concatenate([rng.integers(0, 2, n), rng.integers(0, 2, n)]),
    "gender": ["F"] * n + ["M"] * n,
})

proxy = FairnessAnalyzer.from_dataframe(
    df, y_pred_col="y_pred", sensitive_col="gender", y_true_col="y_true"
)

dpd = proxy.demographic_parity_difference(with_ci=True)
eod = proxy.equalized_odds_difference()
print(dpd.value, dpd.ci)
print(eod.value)
```

### Loading Data (CSV or Parquet)

`load_data` is exported from the top-level `fairpipe` package (there is no
`fairpipe.io` submodule):

```python
from fairpipe import load_data

df = load_data("data.csv")      # CSV
df = load_data("data.parquet")  # Parquet — same API, auto-detected
```

### CLI Usage

When `--threshold` is set, `--metric` is required. Exit `2` means a usage error
(missing metric, bad flags), not a fairness fail.

```bash
# Validate fairness from CSV (create a small sample first if needed)
python - <<'PY'
import pandas as pd
pd.DataFrame({
    "y_true": [0, 1] * 40,
    "y_pred": [0, 1, 1, 0] * 20,
    "group": ["A"] * 40 + ["B"] * 40,
}).to_csv("data.csv", index=False)
PY

fairpipe validate \
    --csv data.csv \
    --y-true y_true \
    --y-pred y_pred \
    --sensitive group \
    --metric demographic_parity_difference \
    --threshold 0.05 \
    --out report.md
```

### REST API Server

Start a local HTTP server with Swagger UI — useful for non-Python stacks or demos:

```bash
pip install fairpipe[api]
fairpipe serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` in your browser, or call the API directly:

```bash
# Three rows per group — below the default min_group_size=30
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "y_pred": [1, 0, 1, 0, 1, 0],
    "sensitive": ["A", "A", "A", "B", "B", "B"],
    "threshold": 0.05
  }'
```

With the default guard, both groups are excluded: the metric `value` is
`null` (undefined) and `n_per_group` is empty. That is the same intentional
behaviour as the Python quickstart above — do not set `"min_group_size": 1`
just to force a number. Send at least 30 rows per group (or raise the field
only when you mean to change the guard) to get a finite disparity.

**Docker:**
```bash
# From the repo root
docker build -t fairpipe-api .
docker run -p 8000:8000 fairpipe-api
# or: docker compose up
```

## Next Steps

- Read the [User Guide](DOCS.md) for comprehensive documentation
- Check out the [API Reference](api.md) for detailed API documentation — including the full REST API endpoint reference
- See the [Integration Guide](integration_guide.md) for CI/CD integration
- LLM fairness evals (counterfactual / refusal / toxicity / BBQ): [LLM Fairness Evaluation](llm_evals_intro.md) — `pip install 'fairpipe[llm]'`; case study notebook needs no API key
- Review [Performance](PERFORMANCE.md) for optimization tips
