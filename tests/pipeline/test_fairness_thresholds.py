# tests/pipeline/test_fairness_thresholds.py
import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import run_detectors


def test_pipeline_fairness_thresholds(tmp_path):
    """
    CI gate: run detectors and assert we don't exceed agreed thresholds.
    For Phase 3, we keep expectations tight but deterministic on a benign dataset.
    """
    # Balanced synthetic data — ensures representation chi2 won't flag
    n = 200
    rng = np.random.default_rng(42)
    g = np.repeat(["A", "B"], n // 2)

    # Benign features (weak association with group) + a target
    f1 = rng.normal(loc=0.0, scale=1.0, size=n)
    f2 = rng.normal(loc=0.0, scale=1.0, size=n)
    y = (f1 + f2 + rng.normal(scale=0.5, size=n) > 0).astype(int)

    df = pd.DataFrame({"g": g, "f1": f1, "f2": f2, "y": y})

    # Inline YAML config (alpha moderately strict; proxy_threshold fairly high)
    cfg_text = """
sensitive: ["g"]
alpha: 0.05
proxy_threshold: 0.6
# (optional) write a report artifact if you want to inspect locally:
# report_out: "artifacts/ci_detector_report.json"
pipeline: []
"""
    cfg = load_config(text=cfg_text)

    report = run_detectors(df, cfg)

    # Gate conditions (Phase 3 default policy):
    #  - No representation flags
    #  - No disparity flags
    #  - No proxy flags
    summary = report.body["summary"]
    assert summary["representation_flags"] == 0, f"Representation flags: {summary}"
    assert summary["disparity_flags"] == 0, f"Disparity flags: {summary}"
    assert summary["proxy_flags"] == 0, f"Proxy flags: {summary}"
