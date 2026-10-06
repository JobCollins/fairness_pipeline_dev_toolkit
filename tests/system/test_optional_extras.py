"""Guards for optional extras (clear install hints)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fairness_pipeline_dev_toolkit.exceptions import DependencyError
from fairness_pipeline_dev_toolkit.integration import mlflow_logger


def test_log_fairness_metrics_requires_tracking_extra():
    """MLflow helpers must name fairpipe[tracking] when mlflow is absent."""
    with patch.object(
        mlflow_logger,
        "_require_mlflow",
        side_effect=DependencyError(
            "MLflow tracking features require the tracking extra. "
            'Install with: pip install "fairpipe[tracking]".',
            dependency_name="mlflow",
            extra_name="tracking",
        ),
    ):
        with pytest.raises(DependencyError) as excinfo:
            mlflow_logger.log_fairness_metrics({"x": {"value": 0.1}})
    text = str(excinfo.value)
    assert "tracking" in text
    assert "mlflow" in text.lower()


def test_require_dependency_message_names_extra():
    from fairness_pipeline_dev_toolkit._extras import require_dependency

    with pytest.raises(DependencyError) as excinfo:
        require_dependency(
            "definitely_not_a_real_module_xyz",
            dependency_name="definitely_not_a_real_module_xyz",
            extra_name="tracking",
            purpose="test purpose",
        )
    assert "fairpipe[tracking]" in str(excinfo.value)
