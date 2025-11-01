from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PipelineStep:
    name: str  # logical name
    transformer: str  # "InstanceReweighting" | "DisparateImpactRemover" | ...
    params: Dict[str, Any] = field(default_factory=dict)  # kwargs to pass to ctor


@dataclass
class PipelineConfig:
    """
    Typed config for the Pipeline Module.
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
        if not isinstance(step, dict):
            continue
        # accept either new 'transformer' or legacy 'kind'
        transformer = step.get("transformer") or step.get("kind")
        if not transformer:
            continue
        steps.append(
            PipelineStep(
                name=str(step.get("name", f"step_{i}")),
                transformer=str(transformer),
                params=dict(step.get("params", {})),
            )
        )
    return steps


def _to_cfg(raw: Dict[str, Any]) -> PipelineConfig:
    # Accept either 'pipeline' (new) or 'steps' (legacy)
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
    profile: Optional[str] = None,
) -> PipelineConfig:
    """
    Load a pipeline config from one of:
      - path=<file.yml>               (string path)
      - text=<yaml string>            (YAML in-memory)
      - obj=<python dict>             (already-parsed mapping)

    Optional:
      - profile=<name>                Select a top-level 'profiles:<name>' entry.

    Exactly one of path/text/obj must be provided. Unknown keys are ignored.

    Behavior:
      - If the root has 'profiles', we select that profile (arg takes precedence,
        then FPDT_PROFILE env var). If not present, default to 'pipeline' if it exists,
        else the sole profile if there's exactly one; otherwise raise with choices.
      - If no 'profiles' key, treat as a flat/legacy config.
    """
    provided = sum(x is not None for x in (path, text, obj))
    if provided != 1:
        raise ValueError("Provide exactly one of: path=, text=, or obj=.")

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
    elif text is not None:
        root = yaml.safe_load(text) or {}
    else:
        root = obj or {}

    if not isinstance(root, dict):
        raise TypeError("Config must parse to a mapping/dict.")

    # Profile-aware mode
    if "profiles" in root and isinstance(root["profiles"], dict):
        profiles: Dict[str, Any] = root.get("profiles") or {}
        chosen = profile or os.getenv("FPDT_PROFILE")

        if not chosen:
            if "pipeline" in profiles:
                chosen = "pipeline"
            elif len(profiles) == 1:
                chosen = next(iter(profiles.keys()))
            else:
                avail = ", ".join(sorted(profiles.keys()))
                raise ValueError(
                    "This config defines profiles, but no profile was selected. "
                    f"Available profiles: {avail}"
                )

        if chosen not in profiles:
            avail = ", ".join(sorted(profiles.keys()))
            raise ValueError(f"Unknown profile '{chosen}'. Available: {avail}")

        flat = profiles[chosen] or {}
        return _to_cfg(flat)

    # Flat/legacy config
    return _to_cfg(root)
