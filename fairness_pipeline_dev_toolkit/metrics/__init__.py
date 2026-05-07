"""
Public API for metrics and fairness analysis.

Public exports:
- FairnessAnalyzer: Main class for computing fairness metrics
- MetricResult: Result object containing metric values and metadata

Internal modules (do not import directly):
- .base: Base classes and interfaces
- .core: Core implementation
- .aequitas_adapter: Adapter implementation (use FairnessAnalyzer instead)
- .fairlearn_adapter: Adapter implementation (use FairnessAnalyzer instead)
"""

from .base import MetricResult
from .core import FairnessAnalyzer as FairnessAnalyzer
from .core import FairnessAnalyzerDataFrameProxy as FairnessAnalyzerDataFrameProxy

__all__ = ["FairnessAnalyzer", "FairnessAnalyzerDataFrameProxy", "MetricResult"]
