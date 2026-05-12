"""Compatibility shim for `fairpipe.api`."""

import fairness_pipeline_dev_toolkit.api as _src
from fairness_pipeline_dev_toolkit.api import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
