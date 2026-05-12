"""Compatibility shim for `fairpipe.stats.bootstrap`."""

import fairness_pipeline_dev_toolkit.stats.bootstrap as _src
from fairness_pipeline_dev_toolkit.stats.bootstrap import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
