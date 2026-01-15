"""Exception hierarchy for fairness toolkit."""


class FairnessToolkitError(Exception):
    """Base exception for all toolkit errors."""


class ConfigValidationError(FairnessToolkitError):
    """Raised when configuration validation fails."""


class MetricComputationError(FairnessToolkitError):
    """Raised when metric computation fails."""


class PipelineExecutionError(FairnessToolkitError):
    """Raised when pipeline execution fails."""


class TrainingError(FairnessToolkitError):
    """Raised when training fails."""
