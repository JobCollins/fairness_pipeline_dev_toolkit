from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import PipelineConfig, PipelineStep
from ..detectors import (
    ProxyVariableDetector,
    RepresentationBiasDetector,
    StatisticalDisparityDetector,
)
from ..detectors.report import BiasReport
from ..transformers.disparate_impact import DisparateImpactRemover
from ..transformers.instance_reweighting import InstanceReweighting


def run_detectors(df: pd.DataFrame, cfg: PipelineConfig) -> BiasReport:  # <-- return BiasReport
    """
    Execute Representation, Disparity, and Proxy detectors for every sensitive attribute.
    Returns a BiasReport (with attribute access via .meta) and writes JSON if cfg.report_out is set.
    """
    rep = RepresentationBiasDetector(alpha=cfg.alpha)
    disp = StatisticalDisparityDetector(alpha=cfg.alpha)
    prox = ProxyVariableDetector(threshold=cfg.proxy_threshold)

    report: Dict[str, Any] = {"summary": {}, "representation": [], "disparities": [], "proxies": []}

    for attr in cfg.sensitive:
        # 1) Representation vs. optional benchmarks
        r = rep.run(df, attr, (cfg.benchmarks or {}).get(attr) if cfg.benchmarks else None)
        report["representation"].append(
            {
                "attribute": r.attribute,
                "counts": r.counts,
                "proportions": r.proportions,
                "benchmark": r.benchmark,
                "chi2_pvalue": r.chi2_pvalue,
                "flagged": r.flagged,
            }
        )

        # 2) Statistical disparities across features
        report["disparities"].extend([dr.__dict__ for dr in disp.run(df, attr)])

        # 3) Proxy variable associations
        report["proxies"].extend([pr.__dict__ for pr in prox.run(df, attr)])

    report["summary"] = {
        "sensitive": cfg.sensitive,
        "alpha": cfg.alpha,
        "proxy_threshold": cfg.proxy_threshold,
        "representation_flags": sum(int(x["flagged"]) for x in report["representation"]),
        "disparity_flags": sum(int(x["flagged"]) for x in report["disparities"]),
        "proxy_flags": sum(int(x["flagged"]) for x in report["proxies"]),
    }

    # Minimal meta so tests can do: rep.meta["phase"] == "0"
    meta = {
        "phase": "0",
        "alpha": cfg.alpha,
        "proxy_threshold": cfg.proxy_threshold,
    }

    bias_report = BiasReport(meta=meta, body=report)

    if cfg.report_out:
        # Persist with meta included at the top level, for clarity in artifacts
        with open(cfg.report_out, "w", encoding="utf-8") as f:
            json.dump(bias_report.to_dict(), f, indent=2, ensure_ascii=False)

    return bias_report


_TRANSFORMER_REGISTRY = {
    "InstanceReweighting": InstanceReweighting,
    "DisparateImpactRemover": DisparateImpactRemover,
}


def _make_step(step: PipelineStep, cfg: PipelineConfig):
    """
    Map config step into an instantiated transformer.
    We inject sensible defaults from cfg when parameters are omitted.
    """
    if step.transformer not in _TRANSFORMER_REGISTRY:
        raise ValueError(f"Unknown transformer '{step.transformer}'.")

    cls = _TRANSFORMER_REGISTRY[step.transformer]
    params = dict(step.params or {})

    # Smart defaults from cfg
    if cls is InstanceReweighting:
        params.setdefault("sensitive", cfg.sensitive)
        params.setdefault("benchmarks", cfg.benchmarks)
    elif cls is DisparateImpactRemover:
        # require a single sensitive attribute for Phase 2
        if "sensitive" not in params:
            if len(cfg.sensitive) != 1:
                raise ValueError(
                    "DisparateImpactRemover requires one 'sensitive' attribute; "
                    "set in step.params or config.sensitive=[one]."
                )
            params["sensitive"] = cfg.sensitive[0]
        if "features" not in params:
            raise ValueError("DisparateImpactRemover step requires 'features': list[str].")

    return (step.name, cls(**params))


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    """
    Build an sklearn Pipeline from cfg.pipeline.
    """
    steps = [_make_step(s, cfg) for s in cfg.pipeline]
    if not steps:
        raise ValueError("Config has no pipeline steps. Add a 'pipeline:' section to your YAML.")
    return Pipeline(steps=steps)


def apply_pipeline(
    pipe: Pipeline, X: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Fit/transform convenience that also returns any auxiliary artifacts from steps:
    - For InstanceReweighting: returns {'sample_weight': np.ndarray}
    - Otherwise: {}
    """
    Xt = pipe.fit_transform(X)
    artifacts: Dict[str, Any] = {}
    for name, step in pipe.steps:
        if isinstance(step, InstanceReweighting):
            artifacts["sample_weight"] = step.sample_weight_
    return Xt, (artifacts or None)
