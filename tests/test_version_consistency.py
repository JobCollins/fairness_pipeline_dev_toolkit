"""Assert packaged version metadata matches the importable ``__version__``."""

from __future__ import annotations

from pathlib import Path

import fairpipe


def _pyproject_version() -> str:
    text = (
        Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(encoding="utf-8")
    )
    for line in text.splitlines():
        if line.startswith("version = "):
            # version = "0.12.0"
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("version = ... not found in pyproject.toml")


def test_pyproject_version_matches_fairpipe_dunder() -> None:
    assert fairpipe.__version__ == _pyproject_version()
