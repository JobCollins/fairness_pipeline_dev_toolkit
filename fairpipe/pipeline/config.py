"""Compatibility shim for ``fairpipe.pipeline.config``.

Re-exports :mod:`fairness_pipeline_dev_toolkit.pipeline.config` so documented
imports such as ``from fairpipe.pipeline.config import PipelineConfig,
find_config_file`` resolve.
"""

import fairness_pipeline_dev_toolkit.pipeline.config as _src
from fairness_pipeline_dev_toolkit.pipeline.config import *  # noqa: F401,F403

__all__ = list(_src.__all__)
