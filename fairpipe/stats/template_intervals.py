"""Compatibility shim for `fairpipe.stats.template_intervals`."""

import fairness_pipeline_dev_toolkit.stats.template_intervals as _src
from fairness_pipeline_dev_toolkit.stats.template_intervals import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
