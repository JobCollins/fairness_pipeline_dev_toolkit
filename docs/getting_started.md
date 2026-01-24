# Getting Started

Welcome to the Fairness Pipeline Development Toolkit! This guide will help you get started quickly.

## Installation

Install the toolkit using pip:

```bash
pip install fairness-pipeline-dev-toolkit
```

For development installation:

```bash
git clone https://github.com/your-org/fairness_pipeline_dev_toolkit.git
cd fairness_pipeline_dev_toolkit
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
import numpy as np

# Create analyzer
fa = FairnessAnalyzer(min_group_size=30, backend="native")

# Your predictions and sensitive attributes
y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
sensitive = np.array(["A", "A", "B", "B", "A", "B", "A", "B"])

# Compute demographic parity difference
result = fa.demographic_parity_difference(y_pred, sensitive)

print(f"Demographic Parity Difference: {result.value:.4f}")
print(f"95% CI: {result.ci}")
```

### CLI Usage

```bash
# Validate fairness from CSV
python -m fairness_pipeline_dev_toolkit.cli.main validate \
    --csv data.csv \
    --y-true y_true \
    --y-pred y_pred \
    --sensitive group \
    --threshold 0.05 \
    --out report.md
```

## Next Steps

- Read the [User Guide](DOCS.md) for comprehensive documentation
- Check out the [API Reference](api.md) for detailed API documentation
- See the [Integration Guide](integration_guide.md) for CI/CD integration
- Review [Performance](PERFORMANCE.md) for optimization tips
