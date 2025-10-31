from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from ..detectors import (
    DetectionReport,
    DisparityDetector,
    ProxyDetector,
    RepresentationDetector,
)
from .registry import TRANSFORMER_REGISTRY


def build_pipeline(cfg: Dict[str, Any]) -> Pipeline:
    """
    Build a sklearn Pipeline from config (Scafold: wiring only).
    Example cfg:
      steps:
        - name: di_repair
          kind: disparate_impact
          params: {columns: ["feature1"], sensitive_col: "group", repair_level: 0.8}
        - name: reweight
          kind: reweighting
          params: {strategy: "none"}
    """
    steps = []
    for step in cfg.get("steps", []):
        kind = step["kind"]
        cls = TRANSFORMER_REGISTRY[kind]
        params = step.get("params", {})
        steps.append((step["name"], cls(**params)))
    return Pipeline(steps)


def run_detectors(df: pd.DataFrame, cfg: Dict[str, Any]) -> DetectionReport:
    """
    Execute detector stubs using config keys:
      sensitive: ["group", ...]
      features: ["x1","x2",...]
      target: "y"
      benchmarks: {group: {A:0.5,B:0.5}}
    """
    sensitive = cfg.get("sensitive", [])
    features = cfg.get("features", [])
    target = cfg.get("target")

    rep = RepresentationDetector().run(df, sensitive=sensitive, benchmarks=cfg.get("benchmarks"))
    disp = DisparityDetector().run(df, target=target, sensitive=sensitive)
    prox = ProxyDetector().run(df, features=features, sensitive=sensitive)
    return DetectionReport(
        representation={"result": rep.__dict__},
        disparity={"result": disp.__dict__},
        proxy={"result": prox.__dict__},
        meta={"phase": "0"},
    )
