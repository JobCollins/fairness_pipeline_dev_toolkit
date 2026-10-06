"""Compatibility shim for `fairpipe.exceptions`."""

from fairness_pipeline_dev_toolkit.exceptions import (  # noqa: F401
    BootstrapUndefinedError,
    ConfigValidationError,
    DataValidationError,
    DependencyError,
    FairnessToolkitError,
    IntervalUndefinedError,
    KamiranCaldersLabelError,
    MetricComputationError,
    PipelineExecutionError,
    TrainingError,
)

__all__ = [
    "FairnessToolkitError",
    "ConfigValidationError",
    "MetricComputationError",
    "PipelineExecutionError",
    "TrainingError",
    "DataValidationError",
    "DependencyError",
    "KamiranCaldersLabelError",
    "IntervalUndefinedError",
    "BootstrapUndefinedError",
]
