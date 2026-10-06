"""Format Wave 3a coverage quotes from committed simulation JSON.

Source of truth for C2b real-data numbers is ``results/tmin_c2b_summary.json``.
Docstrings / CHANGELOG / api.md must quote ``format_c2b_by_T()`` so they cannot
drift from the JSON. Round half-up to 3 decimals (matches the published table).
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Mapping

RESULTS = Path(__file__).resolve().parent / "results"
C2B_SUMMARY = RESULTS / "tmin_c2b_summary.json"
TOXICITY_REALDATA = RESULTS / "toxicity_realdata_summary.json"


def round3(x: float) -> str:
    """Round half-up to three decimal places as a fixed string."""
    d = Decimal(str(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{d:.3f}"


def load_c2b_by_T(path: Path = C2B_SUMMARY) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(T): v for T, v in data["by_T"].items()}


def format_c2b_by_T(
    by_T: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, Dict[str, str]]:
    """Return ``{T: {"mean": "0.929", "worst": "0.920"}}`` from summary JSON."""
    raw = by_T if by_T is not None else load_c2b_by_T()
    out: Dict[str, Dict[str, str]] = {}
    for T, v in raw.items():
        out[str(T)] = {"mean": round3(v["mean"]), "worst": round3(v["min"])}
    return out


def c2b_rst_table(by_T: Mapping[str, Mapping[str, float]] | None = None) -> str:
    quotes = format_c2b_by_T(by_T)
    lines = [
        "=======  ===========  ==========",
        "T        mean cov     worst cov",
        "=======  ===========  ==========",
    ]
    for T in sorted(quotes, key=int):
        q = quotes[T]
        lines.append(f"{T:<7}  {q['mean']:<11}  {q['worst']}")
    lines.append("=======  ===========  ==========")
    return "\n".join(lines)


def c2b_markdown_table(by_T: Mapping[str, Mapping[str, float]] | None = None) -> str:
    quotes = format_c2b_by_T(by_T)
    lines = [
        "| T | mean | worst |",
        "|---|------|-------|",
    ]
    for T in sorted(quotes, key=int):
        q = quotes[T]
        lines.append(f"| {T} | {q['mean']} | {q['worst']} |")
    return "\n".join(lines)


def c2b_changelog_phrase(by_T: Mapping[str, Mapping[str, float]] | None = None) -> str:
    quotes = format_c2b_by_T(by_T)
    parts = []
    for T in sorted(quotes, key=int):
        q = quotes[T]
        if T == "3":
            parts.append(f"T=3 mean {q['mean']} / worst {q['worst']}")
        else:
            parts.append(f"T={T} {q['mean']} / {q['worst']}")
    return "; ".join(parts)


if __name__ == "__main__":
    print(c2b_rst_table())
    print()
    print(c2b_markdown_table())
    print()
    print(c2b_changelog_phrase())
