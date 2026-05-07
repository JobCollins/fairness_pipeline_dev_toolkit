"""Compatibility shim for `fairpipe.training`."""

import fairness_pipeline_dev_toolkit.training as _src
from fairness_pipeline_dev_toolkit.training import *  # noqa: F403

__all__ = list(_src.__all__)
