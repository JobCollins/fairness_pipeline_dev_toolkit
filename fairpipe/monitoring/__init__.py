"""Compatibility shim for `fairpipe.monitoring`."""

import fairness_pipeline_dev_toolkit.monitoring as _src
from fairness_pipeline_dev_toolkit.monitoring import *  # noqa: F403

__all__ = list(_src.__all__)
