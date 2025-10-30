"""
Human-readable reporting utilities.

We expose a single `to_markdown_report(results)` that accepts the same mapping
you would pass to the MLflow logger. Useful for CLI, PR comments, and CI logs.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping


def _fmt_ci(ci):
    if not ci or ci[0] is None or ci[1] is None:
        return "—"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _coerce(val: Any) -> Dict[str, Any]:
    if is_dataclass(val):
        return asdict(val)
    if isinstance(val, dict):
        return val
    return {"value": val}


def to_markdown_report(results: Mapping[str, Any], *, title: str = "Fairness Report") -> str:
    """
    Convert a metrics mapping into a Markdown document with a summary table.

    Returns
    -------
    str
        Markdown content suitable for PR comments, artifacts, or CLI output.
    """
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"# {title}", "", f"_Generated: {ts}_", ""]

    # Summary table
    lines.append("| Metric | Value | CI (95%) | Effect Size | n_per_group |")
    lines.append("|---|---:|---|---:|---|")

    for name, val in results.items():
        item = _coerce(val)
        value = item.get("value", "—")
        if isinstance(value, float):
            value = f"{value:.6f}"
        ci = _fmt_ci(item.get("ci"))
        eff = item.get("effect_size", "—")
        if isinstance(eff, float):
            eff = f"{eff:.6f}"
        n_per_group = item.get("n_per_group")
        n_display = json.dumps(n_per_group) if n_per_group else "—"
        lines.append(f"| `{name}` | {value} | {ci} | {eff} | {n_display} |")

    lines.append("")
    lines.append("> Note: `—` indicates unavailable due to insufficient data or configuration.")
    return "\n".join(lines)
