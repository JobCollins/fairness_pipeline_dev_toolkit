"""
Measurement Module public API.

This facade re-exports the user-facing surface so callers can
`from fairness_pipeline_dev_toolkit.measurement import FairnessAnalyzer, ...`
without depending on internal layout.
"""

from ..integration.mlflow_logger import log_fairness_metrics  # noqa: F401
from ..integration.pytest_plugin import assert_fairness  # noqa: F401
from ..integration.reporting import to_markdown_report  # noqa: F401
from ..metrics.core import FairnessAnalyzer, MetricResult  # noqa: F401

# (Optional) expose common stats utilities
try:
    from ..stats.bayesian import beta_binomial_interval  # noqa: F401
    from ..stats.bootstrap import bootstrap_ci  # noqa: F401
    from ..stats.effect_size import risk_ratio  # noqa: F401
except Exception:  # keep facade resilient if optional deps change
    pass

__all__ = [
    "FairnessAnalyzer",
    "MetricResult",
    "to_markdown_report",
    "log_fairness_metrics",
    "assert_fairness",
    "bootstrap_ci",
    "beta_binomial_interval",
    "risk_ratio",
]
