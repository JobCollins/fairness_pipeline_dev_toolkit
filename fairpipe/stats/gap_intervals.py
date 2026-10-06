"""Compatibility shim for `fairpipe.stats.gap_intervals`."""

import fairness_pipeline_dev_toolkit.stats.gap_intervals as _src
from fairness_pipeline_dev_toolkit.stats.gap_intervals import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
