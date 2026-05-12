"""Compatibility shim for `fairpipe.stats.effect_size`."""

import fairness_pipeline_dev_toolkit.stats.effect_size as _src
from fairness_pipeline_dev_toolkit.stats.effect_size import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
