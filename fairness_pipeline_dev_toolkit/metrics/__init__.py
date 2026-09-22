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
from .input_validation import (
    POSITIVE_LABEL,
    LengthMismatchError,
    MetricInputError,
    MulticlassNotSupportedError,
    NonBinaryEncodingError,
)

__all__ = [
    "FairnessAnalyzer",
    "FairnessAnalyzerDataFrameProxy",
    "MetricResult",
    "MetricInputError",
    "LengthMismatchError",
    "MulticlassNotSupportedError",
    "NonBinaryEncodingError",
    "POSITIVE_LABEL",
]
