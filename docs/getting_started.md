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

```python
from fairpipe.metrics import FairnessAnalyzer
import numpy as np

# Create analyzer
fa = FairnessAnalyzer(min_group_size=30, backend="native")

# Accepts np.ndarray, pd.Series, or list — no .to_numpy() required
y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
sensitive = np.array(["A", "A", "B", "B", "A", "B", "A", "B"])

# Compute demographic parity difference
result = fa.demographic_parity_difference(y_pred, sensitive)

print(f"Demographic Parity Difference: {result.value:.4f}")
print(f"95% CI: {result.ci}")
```

### DataFrame Proxy

If your data is already in a DataFrame, bind a proxy to avoid repeating column names:

```python
import pandas as pd
from fairpipe.metrics import FairnessAnalyzer

df = pd.read_csv("predictions.csv")
proxy = FairnessAnalyzer.from_dataframe(
    df, y_pred_col="y_pred", sensitive_col="gender", y_true_col="y_true"
)

dpd = proxy.demographic_parity_difference(with_ci=True)
eod = proxy.equalized_odds_difference()
```

### Loading Data (CSV or Parquet)

```python
from fairpipe.io import load_data

df = load_data("data.csv")      # CSV
df = load_data("data.parquet")  # Parquet — same API, auto-detected
```

### CLI Usage

```bash
# Validate fairness from CSV
fairpipe validate \
    --csv data.csv \
    --y-true y_true \
    --y-pred y_pred \
    --sensitive group \
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
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "y_pred": [1, 0, 1, 0, 1, 0],
    "sensitive": ["A", "A", "A", "B", "B", "B"],
    "threshold": 0.05,
    "min_group_size": 1
  }'
```

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
