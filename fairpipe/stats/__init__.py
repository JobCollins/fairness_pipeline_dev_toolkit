"""Compatibility shim for `fairpipe.stats`."""

import fairness_pipeline_dev_toolkit.stats as _src
from fairness_pipeline_dev_toolkit.stats import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
