from __future__ import annotations

import io
from typing import Any, Dict

import yaml


def load_config(path: str | None = None, *, text: str | None = None) -> Dict[str, Any]:
    """
    Load YAML config from file path or raw text.
    Scafold: no schema validation (added in later).
    """
    if (path is None) == (text is None):
        raise ValueError("Provide exactly one of `path` or `text`.")
    raw = text if text is not None else open(path, "r", encoding="utf-8").read()
    return yaml.safe_load(io.StringIO(raw)) or {}
