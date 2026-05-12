"""Compatibility shim for `fairpipe.api.app`."""

import fairness_pipeline_dev_toolkit.api.app as _src
from fairness_pipeline_dev_toolkit.api.app import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
