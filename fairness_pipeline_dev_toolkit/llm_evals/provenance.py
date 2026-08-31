"""Attach MetricResult.caveat when the cache's fixture manifest is illustrative."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

DEFAULT_ILLUSTRATIVE_CAVEAT = (
    "Illustrative fixture (BL-009): this value is not evidence of group-level disparity."
)
CAVEAT_RECORDED_REFUSAL = (
    "Demo fixture (BL-009): recorded_refusal is a hiring-cache copy; "
    "this value is not evidence of refusal-rate disparity."
)
CAVEAT_RECORDED_TOXICITY = (
    "Demo fixture (BL-009): recorded_toxicity is a hiring-cache copy; "
    "this value is not evidence of toxicity/sentiment disparity."
)
CAVEAT_RECORDED_BBQ = (
    "Demo fixture (BL-009): recorded_bbq is an all-ambiguous subset; "
    "this value is not evidence of stereotype association."
)


def _manifest_for_cache_dir(cache_dir: Path) -> Optional[Path]:
    for candidate in (cache_dir / "manifest.json", cache_dir.parent / "manifest.json"):
        if candidate.is_file():
            return candidate
    return None


def caveat_for_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """Return a caveat iff the cache's sibling/parent ``manifest.json`` has ``illustrative: true``.

    Closing BL-009 is: re-record into the same paths and set ``illustrative`` false/absent.
    Path identity is not part of the trigger.
    """
    if not cache_dir:
        return None
    try:
        resolved = Path(cache_dir).expanduser().resolve()
    except OSError:
        return None
    manifest_path = _manifest_for_cache_dir(resolved)
    if manifest_path is None:
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("illustrative") is not True:
        return None
    text = payload.get("caveat")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return DEFAULT_ILLUSTRATIVE_CAVEAT


def with_fixture_caveat(metric: MetricResult, cache_dir: Optional[str]) -> MetricResult:
    text = caveat_for_cache_dir(cache_dir)
    if not text:
        return metric
    return replace(metric, caveat=text)
