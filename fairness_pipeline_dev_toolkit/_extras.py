"""Helpers for optional extras (clear install hints instead of raw ImportError)."""

from __future__ import annotations

from typing import Any

from fairness_pipeline_dev_toolkit.exceptions import DependencyError


def require_dependency(
    module_name: str,
    *,
    dependency_name: str,
    extra_name: str,
    purpose: str | None = None,
) -> Any:
    """Import ``module_name`` or raise :class:`DependencyError` naming the extra.

    Parameters
    ----------
    module_name :
        Dotted module to import (e.g. ``"mlflow"``, ``"plotly.graph_objects"``).
    dependency_name :
        Package name shown in the error (e.g. ``"mlflow"``, ``"plotly"``).
    extra_name :
        ``pip install "fairpipe[<extra>]"`` extra name.
    purpose :
        Optional short phrase describing why the dependency is needed.

    Returns
    -------
    module
        The imported module.

    Examples
    --------
    >>> mlflow = require_dependency(  # doctest: +SKIP
    ...     "mlflow", dependency_name="mlflow", extra_name="tracking"
    ... )
    """
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as err:
        why = purpose or f"{dependency_name} is required"
        raise DependencyError(
            f'{why}. Install with: pip install "fairpipe[{extra_name}]".',
            dependency_name=dependency_name,
            extra_name=extra_name,
        ) from err
