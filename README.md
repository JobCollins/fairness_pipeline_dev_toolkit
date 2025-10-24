# Fairness Pipeline Development Toolkit — Measurement Module


A unified, statistically‑rigorous **fairness measurement** layer for ML workflows. Wraps multiple libraries (Fairlearn, Aequitas) behind a single API, computes bootstrap CIs / effect sizes, and integrates with MLflow & pytest/CI.


## Features
- Unified `FairnessAnalyzer` with adapters (Fairlearn/Aequitas)
- Classification & regression fairness metrics, intersectional analysis
- Bootstrap (and small‑n Bayesian) intervals + effect sizes
- MLflow logging, pytest assertions, CLI validation


## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
pytest -q