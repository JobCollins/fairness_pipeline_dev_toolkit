"""Deprecation helpers for packaged LLM fixture accessors (Wave 4 / BL-030)."""

from __future__ import annotations

import warnings

_REMOVAL = (
    "{name} is deprecated and will be removed in the next release. "
    "Packaged fixtures remain available this release only; prefer your own "
    "cache_dir / recorded responses for new code."
)


def warn_fixture_helper_deprecated(name: str, *, stacklevel: int = 2) -> None:
    """Emit :class:`FutureWarning` for a public packaged-fixture helper."""
    warnings.warn(_REMOVAL.format(name=name), FutureWarning, stacklevel=stacklevel)
