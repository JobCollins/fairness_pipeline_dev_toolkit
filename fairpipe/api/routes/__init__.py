"""Compatibility shim for `fairpipe.api.routes`."""

import fairness_pipeline_dev_toolkit.api.routes as _src
from fairness_pipeline_dev_toolkit.api.routes import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
