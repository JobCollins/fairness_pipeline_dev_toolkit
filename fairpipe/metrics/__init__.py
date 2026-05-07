"""Compatibility shim for `fairpipe.metrics`."""

import fairness_pipeline_dev_toolkit.metrics as _src
from fairness_pipeline_dev_toolkit.metrics import *  # noqa: F403

__all__ = list(_src.__all__)
