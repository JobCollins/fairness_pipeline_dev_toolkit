"""
Structured logging infrastructure for the Fairness Pipeline Development Toolkit.

Provides consistent, structured logging across all modules with support for:
- JSON structured logging
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Contextual information (workflow IDs, step names, etc.)
- Performance timing
- Integration with CLI and monitoring modules
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Default log format for structured logging
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_LOG_FORMAT = True  # Use JSON format by default


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add any extra fields
        if hasattr(record, "workflow_id"):
            log_data["workflow_id"] = record.workflow_id
        if hasattr(record, "step"):
            log_data["step"] = record.step
        if hasattr(record, "duration"):
            log_data["duration_seconds"] = record.duration
        if hasattr(record, "n_samples"):
            log_data["n_samples"] = record.n_samples
        if hasattr(record, "metric_name"):
            log_data["metric_name"] = record.metric_name
        if hasattr(record, "group_count"):
            log_data["group_count"] = record.group_count

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True,
    include_console: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        json_format: Whether to use JSON format (True) or plain text (False)
        include_console: Whether to output logs to console

    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger("fairness_pipeline_dev_toolkit")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"fairness_pipeline_dev_toolkit.{name}")


class PerformanceLogger:
    """Context manager for logging performance metrics."""

    def __init__(self, logger: logging.Logger, operation: str, **context: Any):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Name of the operation being timed
            **context: Additional context to include in log
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        import time

        self.start_time = time.time()
        self.logger.info(
            f"Starting {self.operation}", extra={"step": self.operation, **self.context}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        import time

        if self.start_time:
            duration = time.time() - self.start_time
            level = logging.ERROR if exc_type else logging.INFO
            self.logger.log(
                level,
                f"Completed {self.operation}",
                extra={"step": self.operation, "duration": duration, **self.context},
            )
        return False  # Don't suppress exceptions


def log_metric_computation(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    n_samples: int,
    group_count: Optional[int] = None,
    duration: Optional[float] = None,
    **extra: Any,
):
    """
    Log a metric computation with structured information.

    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Computed metric value
        n_samples: Number of samples processed
        group_count: Number of groups (for intersectional analysis)
        duration: Computation duration in seconds
        **extra: Additional context
    """
    logger.info(
        f"Computed {metric_name}: {value:.4f}",
        extra={
            "metric_name": metric_name,
            "metric_value": value,
            "n_samples": n_samples,
            "group_count": group_count,
            "duration": duration,
            **extra,
        },
    )


def log_workflow_step(
    logger: logging.Logger,
    step_name: str,
    workflow_id: Optional[str] = None,
    status: str = "started",
    **context: Any,
):
    """
    Log a workflow step with structured information.

    Args:
        logger: Logger instance
        step_name: Name of the workflow step
        workflow_id: Optional workflow identifier
        status: Step status (started, completed, failed)
        **context: Additional context
    """
    logger.info(
        f"Workflow step {step_name}: {status}",
        extra={"step": step_name, "workflow_id": workflow_id, "status": status, **context},
    )
