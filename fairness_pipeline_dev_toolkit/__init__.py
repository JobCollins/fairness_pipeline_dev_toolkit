"""
Fairness Pipeline Development Toolkit.

A production-ready toolkit for fairness measurement, mitigation, and monitoring
in machine learning pipelines.

Public API:
- Core metrics: Import from fairness_pipeline_dev_toolkit.metrics
- Pipeline utilities: Import from fairness_pipeline_dev_toolkit.pipeline
- Integration tools: Import from fairness_pipeline_dev_toolkit.integration
- Exceptions: Import from fairness_pipeline_dev_toolkit.exceptions

For convenience, commonly used classes are also available at the root level.
However, for better organization, prefer importing from submodules.

Example:
    from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
    from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig
    from fairness_pipeline_dev_toolkit.integration import execute_workflow
"""

from .measurement import (  # noqa: F401
    FairnessAnalyzer,
    MetricResult,
    assert_fairness,
    log_fairness_metrics,
    to_markdown_report,
)

__all__ = [
    "FairnessAnalyzer",
    "MetricResult",
    "to_markdown_report",
    "log_fairness_metrics",
    "assert_fairness",
]

__version__ = "0.5.2"
