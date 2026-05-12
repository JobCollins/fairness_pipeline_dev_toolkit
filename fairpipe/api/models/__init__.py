"""Compatibility shim for `fairpipe.api.models`."""

import fairness_pipeline_dev_toolkit.api.models as _src
from fairness_pipeline_dev_toolkit.api.models import *  # noqa: F403

__all__ = [x for x in dir(_src) if not x.startswith("_")]
