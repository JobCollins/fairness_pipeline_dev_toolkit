"""Compatibility shim for `fairpipe.llm_evals`."""

import fairness_pipeline_dev_toolkit.llm_evals as _src
from fairness_pipeline_dev_toolkit.llm_evals import *  # noqa: F403

__all__ = list(_src.__all__)
