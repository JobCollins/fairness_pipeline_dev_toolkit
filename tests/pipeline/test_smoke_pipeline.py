"""
Enhanced smoke tests for pipeline orchestration.
"""

import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import run_detectors


def test_pipeline_smoke(tmp_path):
    """Test basic pipeline detector execution with enhanced assertions."""
    cfg_text = """
sensitive: ["g"]
features: ["f1","f2"]
target: "y"
steps:
  - name: di
    kind: disparate_impact
    params: {columns: ["f1"], sensitive_col: "g", repair_level: 0.8}
"""
    cfg = load_config(text=cfg_text)
    df = pd.DataFrame({"g": ["A", "B"], "f1": [0.1, 0.2], "f2": [1, 2], "y": [0, 1]})

    # Run detectors
    rep = run_detectors(df, cfg)

    # Enhanced assertions
    assert rep is not None, "Detector report should not be None"
    assert hasattr(rep, "meta"), "Report should have meta attribute"
    assert rep.meta["phase"] == "0", "Report phase should be '0'"
    assert hasattr(rep, "body"), "Report should have body attribute"
    assert isinstance(rep.body, dict), "Report body should be a dictionary"

    # Verify report structure
    assert "summary" in rep.body, "Report body should contain 'summary'"
    summary = rep.body["summary"]
    assert isinstance(summary, dict), "Summary should be a dictionary"

    # Verify expected keys in summary
    expected_keys = ["representation_flags", "disparity_flags", "proxy_flags"]
    for key in expected_keys:
        assert key in summary, f"Summary should contain '{key}'"
        assert isinstance(summary[key], (int, float)), f"'{key}' should be numeric"
