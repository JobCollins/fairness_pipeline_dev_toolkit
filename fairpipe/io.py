"""Compatibility shim for ``fairpipe.io``.

Re-exports :mod:`fairness_pipeline_dev_toolkit.io` so documented
``from fairpipe.io import load_data`` and ``import fairpipe.io`` work.
"""

import fairness_pipeline_dev_toolkit.io as _src
from fairness_pipeline_dev_toolkit.io import *  # noqa: F401,F403

__all__ = list(getattr(_src, "__all__", ["load_data"]))
