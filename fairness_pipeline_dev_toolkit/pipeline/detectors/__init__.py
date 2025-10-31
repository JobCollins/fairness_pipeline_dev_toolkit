"""Bias Detection Engine (Phase 0 stubs)."""

from .core import (
    DisparityResult,
    ProxyResult,
    ProxyVariableDetector,
    RepresentationBiasDetector,
    RepresentationResult,
    StatisticalDisparityDetector,
)
from .report import DetectionReport

__all__ = [
    "RepresentationBiasDetector",
    "StatisticalDisparityDetector",
    "ProxyVariableDetector",
    "RepresentationResult",
    "DisparityResult",
    "ProxyResult",
    "DetectionReport",
]
