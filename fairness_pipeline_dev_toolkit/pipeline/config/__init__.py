from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PipelineStep:
    name: str  # logical name
    transformer: str  # "InstanceReweighting" | "DisparateImpactRemover"
    params: Dict[str, Any] = field(default_factory=dict)  # kwargs to pass to ctor


@dataclass
class PipelineConfig:
    """
    Typed config for the Pipeline Module. Keep it minimal in Phase 1:
    - sensitive: list of column names to treat as protected attributes
    - benchmarks: optional expected proportions per attribute (for representation checks)
    - alpha: significance level for statistical tests (chi-square / ANOVA)
    - proxy_threshold: association threshold (Cramér's V / eta²) to flag proxies
    - report_out: optional path to write the JSON report
    - pipeline: ordered list of steps to instantiate into an sklearn Pipeline
    """

    sensitive: List[str]
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None
    alpha: float = 0.05
    proxy_threshold: float = 0.30
    report_out: Optional[str] = None
    pipeline: List[PipelineStep] = field(default_factory=list)


def _parse_steps(raw_steps: Optional[List[dict]]) -> List[PipelineStep]:
    steps: List[PipelineStep] = []
    if not raw_steps:
        return steps
    for i, step in enumerate(raw_steps):
        if not isinstance(step, dict) or "transformer" not in step:
            continue
        # Prefer new key; fall back to legacy 'kind'
        transformer = step.get("transformer") or step.get("kind")
        if not transformer:
            # Skip malformed entries but keep robust
            continue
        steps.append(
            PipelineStep(
                name=str(step.get("name", f"step_{i}")),
                transformer=str(step["transformer"]),
                params=dict(step.get("params", {})),
            )
        )
    return steps


def _to_cfg(raw: Dict[str, Any]) -> PipelineConfig:
    # Backwards-compat: accept either 'pipeline' (new) or 'steps' (legacy)
    raw_steps = raw.get("pipeline")
    if raw_steps is None:
        raw_steps = raw.get("steps")

    return PipelineConfig(
        sensitive=list(raw.get("sensitive", [])),
        benchmarks=raw.get("benchmarks"),
        alpha=float(raw.get("alpha", 0.05)),
        proxy_threshold=float(raw.get("proxy_threshold", 0.30)),
        report_out=raw.get("report_out"),
        pipeline=_parse_steps(raw_steps),
    )


def load_config(
    path: Optional[str] = None,
    *,
    text: Optional[str] = None,
    obj: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """
    Load a pipeline config from one of:
      - path=<file.yml>               (string path)
      - text=<yaml string>            (YAML in-memory)
      - obj=<python dict>             (already-parsed mapping)

    Exactly one source must be provided. Unknown keys are ignored.
    """
    provided = sum(x is not None for x in (path, text, obj))
    if provided != 1:
        raise ValueError("Provide exactly one of: path=, text=, or obj=.")

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif text is not None:
        raw = yaml.safe_load(text) or {}
    else:
        raw = obj or {}

    if not isinstance(raw, dict):
        raise TypeError("Config must parse to a mapping/dict.")

    return _to_cfg(raw)
