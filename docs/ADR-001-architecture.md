# ADR-001: Measurement Module Architecture

**Context:** Standardize fairness measurement across teams with defensible stats and smooth integration.

**Decision:**
- Single entry point: `FairnessAnalyzer` (adapters for Fairlearn/Aequitas)
- Stats layer: bootstrap CIs (default), Bayesian for small-n (later)
- Outputs: structured schema {metric, value, ci, effect_size, n_per_group}
- Integration: MLflow logging + pytest assertion helpers + CI gates

**Status:**