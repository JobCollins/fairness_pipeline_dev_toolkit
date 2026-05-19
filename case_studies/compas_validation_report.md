# Fairness Validation Report (CLI)

_Generated: 2026-05-19 16:27:45 UTC_

| Metric | Value | CI (95%) | Effect Size | n_per_group |
|---|---:|---|---:|---|
| `demographic_parity_difference` | 0.245107 | [0.2184, 0.2699] | 1.740604 | {"African-American": 3175, "Caucasian": 2103} |
| `equalized_odds_difference` | 0.211582 | [0.1881, 0.2493] | 1.923234 | {"African-American": 3175, "Caucasian": 2103} |

> Note: `—` indicates unavailable due to insufficient data or configuration.

## Threshold verdict

- **Metric**: `equalized_odds_difference`
- **Value**: 0.211582
- **Threshold**: 0.050000
- **Rule**: pass if `abs(metric) <= threshold` (same convention as workflow validation).
- **Result**: **FAIL**
