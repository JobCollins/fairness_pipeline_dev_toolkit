import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import run_detectors


def test_pipeline_smoke(tmp_path):
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
    # pipe = build_pipeline(cfg)
    df = pd.DataFrame({"g": ["A", "B"], "f1": [0.1, 0.2], "f2": [1, 2], "y": [0, 1]})
    rep = run_detectors(df, cfg)
    assert rep.meta["phase"] == "0"
