"""Compatibility shim for `fairpipe.pipeline`."""

import fairness_pipeline_dev_toolkit.pipeline as _src
from fairness_pipeline_dev_toolkit.pipeline import *  # noqa: F403

__all__ = list(_src.__all__)
