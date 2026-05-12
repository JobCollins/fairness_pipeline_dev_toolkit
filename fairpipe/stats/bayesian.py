"""Compatibility shim for `fairpipe.stats.bayesian`."""

import fairness_pipeline_dev_toolkit.stats.bayesian as _src
from fairness_pipeline_dev_toolkit.stats.bayesian import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
