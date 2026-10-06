# COMPAS Fairness Validation Report

_Generated: 2026-10-06 15:00:07 UTC_

| Metric | Value | CI (95%) | Effect Size | p_value | n_per_group |
|---|---:|---|---:|---:|---|
| `demographic_parity_difference` | 0.245107 | [0.2185, 0.2713] | 1.740604 | 0.0004998 | {"African-American": 3175, "Caucasian": 2103} |
| `equalized_odds_difference` | 0.211582 | [0.1651, 0.2576] | 1.923234 | 0.0004998 | {"African-American": 3175, "Caucasian": 2103} |

> Note: when a CI is undefined, the CI column shows the reason (e.g. “no calibrated interval for this metric yet, see #63”), never `None`. `—` means the field was not computed. Use `p_value` for significance claims.