"""Compatibility shim for `fairpipe.integration`."""

import fairness_pipeline_dev_toolkit.integration as _src
from fairness_pipeline_dev_toolkit.integration import *  # noqa: F403

__all__ = list(_src.__all__)
