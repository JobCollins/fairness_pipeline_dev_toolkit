"""Compatibility shim for `fairpipe.measurement`."""

import fairness_pipeline_dev_toolkit.measurement as _src
from fairness_pipeline_dev_toolkit.measurement import *  # noqa: F403

__all__ = list(_src.__all__)
