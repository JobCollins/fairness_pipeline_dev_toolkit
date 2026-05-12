"""Compatibility shim for `fairpipe.stats.multipletests`."""

import fairness_pipeline_dev_toolkit.stats.multipletests as _src
from fairness_pipeline_dev_toolkit.stats.multipletests import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
