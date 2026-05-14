"""
Public API for pipeline utilities.

Public exports:
- PipelineConfig, load_config
- build_pipeline, apply_pipeline, run_detectors, PipelineResult
- InstanceReweighting, DisparateImpactRemover, ReweighingTransformer, ProxyDropper

Internal modules (do not import directly):
- .orchestration.registry
- .detectors.core
"""

from .config import PipelineConfig, load_config
from .orchestration import PipelineResult, apply_pipeline, build_pipeline, run_detectors
from .transformers import (
    DisparateImpactRemover,
    InstanceReweighting,
    ProxyDropper,
    ReweighingTransformer,
)

__all__ = [
    "PipelineConfig",
    "load_config",
    "build_pipeline",
    "apply_pipeline",
    "PipelineResult",
    "run_detectors",
    "InstanceReweighting",
    "DisparateImpactRemover",
    "ReweighingTransformer",
    "ProxyDropper",
]
