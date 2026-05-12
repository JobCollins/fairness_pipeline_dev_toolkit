"""Compatibility shim for `fairpipe.integration.reporting`."""

import fairness_pipeline_dev_toolkit.integration.reporting as _src
from fairness_pipeline_dev_toolkit.integration.reporting import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
