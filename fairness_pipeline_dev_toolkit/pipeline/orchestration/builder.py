# fairness_pipeline_dev_toolkit/pipeline/orchestration/builder.py  (EXPANDED lightly)
from __future__ import annotations

from typing import Tuple

from sklearn.pipeline import Pipeline

from ..config import PipelineConfig, PipelineStep
from .registry import get_transformer_class


def _make_step(step: PipelineStep, cfg: PipelineConfig) -> Tuple[str, object]:
    """
    Instantiate a transformer from a PipelineStep using the registry.
    We may inject config-derived defaults here if needed.
    """
    cls = get_transformer_class(step.transformer)
    params = dict(step.params or {})

    # Provide smart defaults commonly used across transformers
    if step.transformer in ("InstanceReweighting", "ReweighingTransformer"):
        params.setdefault("sensitive", cfg.sensitive)
        params.setdefault("benchmarks", cfg.benchmarks)

    if step.transformer == "ProxyDropper":
        params.setdefault("sensitive", cfg.sensitive)
        # If no features is provided, the transformer will use all non-sensitive columns.

    return (step.name, cls(**params))


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    steps = [_make_step(s, cfg) for s in cfg.pipeline]
    if not steps:
        raise ValueError("Config has no pipeline steps. Add a 'pipeline:' section.")
    return Pipeline(steps=steps)
