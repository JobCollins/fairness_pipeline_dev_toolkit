"""BBQ loader — local fixture by default; optional fetch from a pinned upstream commit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

BBQ_UPSTREAM_REPO = "https://github.com/nyu-mll/BBQ"
BBQ_PINNED_COMMIT = "bea11bd97d79217245b5871acd247b9d6eb24598"
BBQ_LICENSE = "CC BY 4.0"
DEFAULT_LOCAL_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "bbq" / "gender_identity_subset.json"
)


def load_bbq_items(
    path: Optional[str | Path] = None,
    *,
    fetch_upstream: bool = False,
    category_file: str = "data/Gender_identity.jsonl",
    max_items: int = 24,
) -> List[Dict[str, Any]]:
    """
    Load BBQ-schema items.

    Default CI path uses the packaged local subset (no network). Set
    ``fetch_upstream=True`` to pull JSONL from the pinned BBQ commit.
    """
    if fetch_upstream:
        url = (
            f"https://raw.githubusercontent.com/nyu-mll/BBQ/" f"{BBQ_PINNED_COMMIT}/{category_file}"
        )
        with urlopen(url, timeout=30) as response:  # nosec B310 — pinned HTTPS GitHub raw
            lines = response.read().decode("utf-8").splitlines()
        items = [json.loads(line) for line in lines if line.strip()]
        return items[:max_items]

    target = Path(path) if path is not None else DEFAULT_LOCAL_FIXTURE
    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return list(payload["items"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognized BBQ fixture format at {target}")


def item_to_prompt(item: Dict[str, Any]) -> str:
    context = item.get("context") or ""
    question = item.get("question") or ""
    answers = [item.get("ans0", ""), item.get("ans1", ""), item.get("ans2", "")]
    labeled = " ".join(f"({idx}) {ans}" for idx, ans in enumerate(answers) if ans)
    return f"{context}\n{question}\nOptions: {labeled}\nReply with 0, 1, or 2."


def stereotypical_answer_index(item: Dict[str, Any]) -> int:
    """Index of the stereotyped option — not the BBQ gold 'unknown' label."""
    additional = item.get("additional_metadata") or {}
    stereotyped = additional.get("stereotyped_groups") or []
    answers = [item.get("ans0", ""), item.get("ans1", ""), item.get("ans2", "")]
    for idx, ans in enumerate(answers):
        tokens = {t.strip(".,;:!?()[]").lower() for t in str(ans).split()}
        if any(str(g).lower() in tokens for g in stereotyped):
            return idx
    label = item.get("label")
    if isinstance(label, int):
        return label
    return 0
