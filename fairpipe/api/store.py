"""Compatibility shim for `fairpipe.api.store`."""

import fairness_pipeline_dev_toolkit.api.store as _src
from fairness_pipeline_dev_toolkit.api.store import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
