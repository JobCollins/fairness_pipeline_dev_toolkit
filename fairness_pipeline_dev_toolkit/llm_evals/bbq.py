"""BBQ loader — explicit path or opt-in upstream fetch; no silent default subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError

BBQ_UPSTREAM_REPO = "https://github.com/nyu-mll/BBQ"
BBQ_PINNED_COMMIT = "bea11bd97d79217245b5871acd247b9d6eb24598"
BBQ_LICENSE = "CC BY 4.0"
# Schema-shaped fixture for tests / recorded-cache helpers only — not a default load.
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

    Requires an explicit ``path`` or ``fetch_upstream=True``. There is no silent
    default item set: the packaged schema fixture under ``fixtures/bbq/`` is for
    tests and recorded-cache helpers only (pass that path explicitly if needed).

    Set ``fetch_upstream=True`` to pull JSONL from the pinned BBQ commit (network).
    """
    if fetch_upstream:
        url = (
            f"https://raw.githubusercontent.com/nyu-mll/BBQ/" f"{BBQ_PINNED_COMMIT}/{category_file}"
        )
        with urlopen(url, timeout=30) as response:  # nosec B310 — pinned HTTPS GitHub raw
            lines = response.read().decode("utf-8").splitlines()
        items = [json.loads(line) for line in lines if line.strip()]
        return items[:max_items]

    if path is None:
        raise ConfigValidationError(
            "load_bbq_items requires path=... or fetch_upstream=True. "
            "There is no default BBQ item set; pass an explicit JSON path "
            "(tests may use fairness_pipeline_dev_toolkit.llm_evals.bbq.DEFAULT_LOCAL_FIXTURE)."
        )

    target = Path(path)
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
    """Index of the stereotyped option — not the BBQ gold 'unknown' label.

    Answer mapping is incomplete (possessives and upstream F/M codes often miss);
    see GitHub issue #31. Prefer treating results as experimental until that lands.
    """
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
