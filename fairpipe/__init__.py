"""
Compatibility shim package.

`fairpipe` re-exports the public API from `fairness_pipeline_dev_toolkit` to preserve
backwards-compatible imports while keeping object identity intact.
"""

import fairness_pipeline_dev_toolkit as _src
from fairness_pipeline_dev_toolkit import *  # noqa: F403
from fairness_pipeline_dev_toolkit.io import load_data  # noqa: F401

__all__ = list(_src.__all__) + ["load_data"]
__version__ = _src.__version__
