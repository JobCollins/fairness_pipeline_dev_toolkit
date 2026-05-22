"""
Public API for integration utilities.

Public exports:
- execute_workflow (runtime: class_weight, decision_threshold), WorkflowResult, ValidationResult
- log_workflow_results, log_fairness_metrics
- to_markdown_report
- assert_fairness

Internal modules (do not import directly):
- .orchestrator (implementation details may change)
- .mlflow_logger
"""

from .mlflow_logger import log_fairness_metrics, log_workflow_results
from .orchestrator import ValidationResult, WorkflowResult, execute_workflow
from .pytest_plugin import assert_fairness
from .reporting import to_markdown_report

__all__ = [
    "execute_workflow",
    "WorkflowResult",
    "ValidationResult",
    "log_workflow_results",
    "log_fairness_metrics",
    "to_markdown_report",
    "assert_fairness",
]
