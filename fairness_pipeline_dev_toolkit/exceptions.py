"""Exception hierarchy for fairness toolkit.

All exceptions inherit from FairnessToolkitError and provide structured
error information with user-friendly messages.
"""

from typing import Any, Dict, Optional


class FairnessToolkitError(Exception):
    """Base exception for all toolkit errors.

    Attributes:
        message: Human-readable error message
        context: Optional context dictionary with additional error details
        suggestion: Optional suggestion for how to fix the error
    """

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize exception with message and optional context.

        Args:
            message: Human-readable error message
            context: Optional dictionary with additional error details
            suggestion: Optional suggestion for how to fix the error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return formatted error message with context and suggestion."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        return " | ".join(parts)


class ConfigValidationError(FairnessToolkitError):
    """Raised when configuration validation fails.

    Use this exception when configuration files or parameters are invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        field_value: Any = None,
        config_path: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize configuration validation error.

        Args:
            message: Error message describing the validation failure
            field_name: Name of the invalid configuration field
            field_value: Value that failed validation
            config_path: Path to the configuration file (if applicable)
            suggestion: Optional suggestion for fixing the configuration
        """
        context = {}
        if field_name:
            context["field"] = field_name
        if field_value is not None:
            context["value"] = str(field_value)
        if config_path:
            context["config_path"] = config_path

        if not suggestion and field_name:
            suggestion = f"Check the '{field_name}' field in your configuration file."

        super().__init__(message, context=context, suggestion=suggestion)


class MetricComputationError(FairnessToolkitError):
    """Raised when metric computation fails.

    Use this exception when fairness metrics cannot be computed due to
    insufficient data, invalid inputs, or computational errors.
    """

    def __init__(
        self,
        message: str,
        *,
        metric_name: Optional[str] = None,
        min_group_size: Optional[int] = None,
        actual_group_sizes: Optional[Dict[str, int]] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize metric computation error.

        Args:
            message: Error message describing the computation failure
            metric_name: Name of the metric that failed to compute
            min_group_size: Required minimum group size
            actual_group_sizes: Actual group sizes (if available)
            suggestion: Optional suggestion for fixing the issue
        """
        context = {}
        if metric_name:
            context["metric"] = metric_name
        if min_group_size is not None:
            context["min_group_size"] = min_group_size
        if actual_group_sizes:
            context["group_sizes"] = actual_group_sizes

        if not suggestion:
            if min_group_size and actual_group_sizes:
                small_groups = [
                    f"{group}({size})"
                    for group, size in actual_group_sizes.items()
                    if size < min_group_size
                ]
                if small_groups:
                    suggestion = (
                        f"Increase data size or reduce min_group_size. "
                        f"Groups below threshold: {', '.join(small_groups)}"
                    )
            elif min_group_size:
                suggestion = (
                    f"Ensure all groups have at least {min_group_size} samples, "
                    f"or reduce the min_group_size parameter."
                )

        super().__init__(message, context=context, suggestion=suggestion)


class PipelineExecutionError(FairnessToolkitError):
    """Raised when pipeline execution fails.

    Use this exception when pipeline transformations or operations fail.
    """

    def __init__(
        self,
        message: str,
        *,
        step_name: Optional[str] = None,
        step_index: Optional[int] = None,
        transformer_name: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize pipeline execution error.

        Args:
            message: Error message describing the execution failure
            step_name: Name of the pipeline step that failed
            step_index: Index of the failed step (0-based)
            transformer_name: Name of the transformer that failed
            suggestion: Optional suggestion for fixing the issue
        """
        context = {}
        if step_name:
            context["step"] = step_name
        if step_index is not None:
            context["step_index"] = step_index
        if transformer_name:
            context["transformer"] = transformer_name

        if not suggestion and step_name:
            suggestion = (
                f"Check the configuration for step '{step_name}' and verify input data format."
            )

        super().__init__(message, context=context, suggestion=suggestion)


class TrainingError(FairnessToolkitError):
    """Raised when training fails.

    Use this exception when model training operations fail.
    """

    def __init__(
        self,
        message: str,
        *,
        method: Optional[str] = None,
        training_params: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize training error.

        Args:
            message: Error message describing the training failure
            method: Training method that failed (e.g., 'reductions', 'lagrangian')
            training_params: Training parameters that were used
            suggestion: Optional suggestion for fixing the issue
        """
        context = {}
        if method:
            context["method"] = method
        if training_params:
            context["params"] = str(training_params)

        if not suggestion and method:
            if method == "reductions":
                suggestion = "Ensure fairlearn is installed: pip install fairness-pipeline-dev-toolkit[adapters]"
            elif method in ("regularized", "lagrangian"):
                suggestion = "Ensure PyTorch is installed: pip install fairness-pipeline-dev-toolkit[training]"

        super().__init__(message, context=context, suggestion=suggestion)


class DataValidationError(FairnessToolkitError):
    """Raised when data validation fails.

    Use this exception when input data does not meet requirements
    (missing columns, wrong types, insufficient samples, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        missing_columns: Optional[list] = None,
        invalid_columns: Optional[Dict[str, str]] = None,
        data_shape: Optional[tuple] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize data validation error.

        Args:
            message: Error message describing the validation failure
            missing_columns: List of required columns that are missing
            invalid_columns: Dictionary mapping column names to validation errors
            data_shape: Shape of the data (rows, columns)
            suggestion: Optional suggestion for fixing the issue
        """
        context = {}
        if missing_columns:
            context["missing_columns"] = missing_columns
        if invalid_columns:
            context["invalid_columns"] = invalid_columns
        if data_shape:
            context["data_shape"] = data_shape

        if not suggestion:
            if missing_columns:
                suggestion = (
                    f"Ensure the following columns are present: {', '.join(missing_columns)}"
                )
            elif invalid_columns:
                suggestion = "Check column types and values match expected formats."

        super().__init__(message, context=context, suggestion=suggestion)


class DependencyError(FairnessToolkitError):
    """Raised when required optional dependencies are missing.

    Use this exception when optional dependencies (torch, fairlearn, etc.)
    are required but not installed.
    """

    def __init__(
        self,
        message: str,
        *,
        dependency_name: Optional[str] = None,
        extra_name: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize dependency error.

        Args:
            message: Error message describing the missing dependency
            dependency_name: Name of the missing dependency (e.g., 'torch', 'fairlearn')
            extra_name: Name of the extra that provides the dependency (e.g., 'training', 'adapters')
            suggestion: Optional suggestion for installing the dependency
        """
        context = {}
        if dependency_name:
            context["dependency"] = dependency_name
        if extra_name:
            context["extra"] = extra_name

        if not suggestion:
            if extra_name:
                suggestion = (
                    f"Install with: pip install fairness-pipeline-dev-toolkit[{extra_name}]"
                )
            elif dependency_name:
                suggestion = f"Install with: pip install {dependency_name}"
            else:
                suggestion = "Check the installation instructions for required dependencies."

        super().__init__(message, context=context, suggestion=suggestion)
