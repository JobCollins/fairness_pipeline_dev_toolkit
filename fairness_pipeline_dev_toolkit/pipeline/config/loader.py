# fairness_pipeline_dev_toolkit/pipeline/config/loader.py
from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError


@dataclass
class PipelineStep:
    name: str  # logical name
    transformer: str  # "InstanceReweighting" | "DisparateImpactRemover" | ...
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    method: str  # "reductions" | "regularized" | "lagrangian"
    target_column: str  # e.g., "y" or "y_true"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    sensitive: List[str]
    features: Optional[List[str]] = None
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None
    alpha: float = 0.05
    proxy_threshold: float = 0.30
    report_out: Optional[str] = None
    pipeline: List[PipelineStep] = field(default_factory=list)
    training: Optional[TrainingConfig] = None
    fairness_metric: Optional[str] = (
        None  # e.g., "demographic_parity_difference", "equalized_odds_difference"
    )
    validation_threshold: Optional[float] = (
        None  # maximum allowed value for primary fairness metric
    )


def _ensure_list_of_strings(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigValidationError(f"Config field '{field_name}' must be a list of strings.")
    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(
                f"Config field '{field_name}' expects non-empty string entries; got {item!r}."
            )
        cleaned.append(item.strip())
    if not cleaned:
        raise ConfigValidationError(f"Config field '{field_name}' must not be empty.")
    return cleaned


def _validate_benchmarks(benchmarks: Any) -> None:
    if benchmarks is None:
        return
    if not isinstance(benchmarks, dict):
        raise ConfigValidationError(
            "Config field 'benchmarks' must be a mapping of attr -> values."
        )
    for attr, group_map in benchmarks.items():
        if not isinstance(attr, str):
            raise ConfigValidationError("Benchmark keys must be attribute names (strings).")
        if not isinstance(group_map, dict):
            raise ConfigValidationError(
                f"Benchmark entry for '{attr}' must be a mapping of group -> proportion."
            )
        for group, value in group_map.items():
            if not isinstance(group, str):
                raise ConfigValidationError(
                    f"Benchmark group names must be strings; got {group!r} under '{attr}'."
                )
            try:
                float(value)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"Benchmark value for '{attr}.{group}' must be numeric; got {value!r}."
                ) from None


def _validate_pipeline_steps(raw_steps: Any) -> None:
    if raw_steps is None:
        return
    if not isinstance(raw_steps, list):
        raise ConfigValidationError("Config field 'pipeline' must be a list of step mappings.")
    for idx, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise ConfigValidationError(f"Pipeline step #{idx + 1} must be a mapping.")
        transformer = step.get("transformer") or step.get("kind")
        if not isinstance(transformer, str) or not transformer.strip():
            raise ConfigValidationError(
                f"Pipeline step #{idx + 1} requires a 'transformer' string."
            )
        params = step.get("params")
        if params is not None and not isinstance(params, dict):
            raise ConfigValidationError(
                f"Pipeline step '{transformer}' params must be a mapping if provided."
            )


def _validate_training_config(training: Any) -> None:
    if training is None:
        return
    if not isinstance(training, dict):
        raise ConfigValidationError("Config field 'training' must be a mapping.")
    method = training.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ConfigValidationError(
            "Config field 'training.method' must be a non-empty string. "
            "Valid values: 'reductions', 'regularized', 'lagrangian'."
        )
    if method not in ("reductions", "regularized", "lagrangian"):
        raise ConfigValidationError(
            f"Config field 'training.method' must be one of: 'reductions', 'regularized', 'lagrangian'. Got: {method!r}."
        )
    target_column = training.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ConfigValidationError(
            "Config field 'training.target_column' must be a non-empty string."
        )
    params = training.get("params")
    if params is not None and not isinstance(params, dict):
        raise ConfigValidationError("Config field 'training.params' must be a mapping if provided.")


def _validate_flat_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return sanitized fields and raise informative errors if schema mismatches."""
    cleaned = dict(raw or {})

    sensitive = cleaned.get("sensitive")
    cleaned["sensitive"] = _ensure_list_of_strings(sensitive or [], "sensitive")

    _validate_benchmarks(cleaned.get("benchmarks"))
    _validate_pipeline_steps(cleaned.get("pipeline") or cleaned.get("steps"))
    _validate_training_config(cleaned.get("training"))

    for float_field in ("alpha", "proxy_threshold", "validation_threshold"):
        if float_field not in cleaned or cleaned[float_field] is None:
            continue
        value = cleaned[float_field]
        try:
            cleaned[float_field] = float(value)
        except (TypeError, ValueError):
            raise ConfigValidationError(
                f"Config field '{float_field}' must be numeric; got {value!r}."
            ) from None

    report_out = cleaned.get("report_out")
    if report_out is not None and not isinstance(report_out, str):
        raise ConfigValidationError(
            "Config field 'report_out' must be a string path when provided."
        )

    fairness_metric = cleaned.get("fairness_metric")
    if fairness_metric is not None and not isinstance(fairness_metric, str):
        raise ConfigValidationError(
            "Config field 'fairness_metric' must be a string when provided."
        )

    features = cleaned.get("features")
    if features is not None:
        cleaned["features"] = _ensure_list_of_strings(features, "features")

    return cleaned


def _parse_steps(raw_steps: Optional[List[dict]]) -> List[PipelineStep]:
    steps: List[PipelineStep] = []
    if not raw_steps:
        return steps
    for i, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            continue
        # accept either 'transformer' (current) or legacy 'kind'
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


def _parse_training_config(raw_training: Optional[Dict[str, Any]]) -> Optional[TrainingConfig]:
    """Parse training config from raw dict."""
    if raw_training is None:
        return None
    return TrainingConfig(
        method=str(raw_training.get("method", "")),
        target_column=str(raw_training.get("target_column", "")),
        params=dict(raw_training.get("params", {})),
    )


def _as_cfg(raw: Dict[str, Any]) -> PipelineConfig:
    """Convert a flat raw dict (already resolved for a profile) into PipelineConfig."""
    raw = _validate_flat_config(raw)
    # accept either 'pipeline' (current) or legacy 'steps'
    raw_steps = raw.get("pipeline")
    if raw_steps is None:
        raw_steps = raw.get("steps")

    return PipelineConfig(
        sensitive=list(raw.get("sensitive", [])),
        features=raw.get("features"),
        benchmarks=raw.get("benchmarks"),
        alpha=float(raw.get("alpha", 0.05)),
        proxy_threshold=float(raw.get("proxy_threshold", 0.30)),
        report_out=raw.get("report_out"),
        pipeline=_parse_steps(raw_steps),
        training=_parse_training_config(raw.get("training")),
        fairness_metric=raw.get("fairness_metric"),
        validation_threshold=raw.get("validation_threshold"),
    )


def _shallow_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge two dicts with override winning."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        out[k] = v
    return out


def find_config_file(
    explicit_path: Optional[str] = None,
    profile: Optional[str] = None,
) -> Optional[Path]:
    """
    Discover configuration file in order of precedence:
    1. explicit_path (if provided)
    2. .fairpipe/config.yml (project-local)
    3. ~/.fairpipe/config.yml (user-global)
    4. None (not found)

    Args:
        explicit_path: Explicit path to config file (highest priority)
        profile: Profile name (not used for discovery, but kept for API consistency)

    Returns:
        Path to config file if found, None otherwise
    """
    # 1. Explicit path (highest priority)
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path.resolve()
        return None

    # 2. Project-local: .fairpipe/config.yml
    # Search from current directory up to filesystem root
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        config_path = parent / ".fairpipe" / "config.yml"
        if config_path.exists():
            return config_path.resolve()

    # 3. User-global: ~/.fairpipe/config.yml
    home_config = Path.home() / ".fairpipe" / "config.yml"
    if home_config.exists():
        return home_config.resolve()

    # 4. Not found
    return None


def _auto_choose_profile(profiles: Dict[str, Any]) -> Optional[str]:
    """
    Robust auto-detection under pytest:
      - parse PYTEST_CURRENT_TEST to find 'tests/<suite>/...' and map suite -> profile
      - if only one profile exists, use it
    """
    ptest = os.environ.get("PYTEST_CURRENT_TEST", "")
    if ptest:
        # e.g. 'tests/pipeline/test_transformers_smoke.py::test... (call)'
        test_path = ptest.split(" ")[0]  # keep left of the first space
        norm = osp.normpath(test_path)
        parts = norm.split(osp.sep)
        if "tests" in parts:
            i = parts.index("tests")
            # suite is the immediate subdir under tests/ (pipeline, training, etc.)
            if i + 1 < len(parts):
                suite = parts[i + 1]
                if suite in profiles:
                    return suite
        # Last-resort while running under pytest: prefer 'pipeline' if present
        if "pipeline" in profiles:
            return "pipeline"

    # No pytest / couldn’t infer — if exactly one profile exists, use it
    if len(profiles) == 1:
        return next(iter(profiles.keys()))
    return None


def load_config(
    path: Optional[str] = None,
    *,
    text: Optional[str] = None,
    obj: Optional[Dict[str, Any]] = None,
    profile: Optional[str] = None,
) -> PipelineConfig:
    """
    Load YAML config with optional 'profiles' support.

    Configuration file discovery (when path=None and no text/obj provided):
      1. explicit_path (if provided)
      2. .fairpipe/config.yml (project-local, searched from current directory up)
      3. ~/.fairpipe/config.yml (user-global)
      4. None (raises error if no text/obj provided)

    Resolution order when 'profiles' is present:
      1) explicit 'profile' argument
      2) env var FPDT_PROFILE
      3) auto-detect by test path (pytest): 'training' for tests/training, 'pipeline' for tests/pipeline
      4) if exactly one profile exists, use it
      5) otherwise, raise a helpful error

    When a profile is chosen, it is shallow-merged over top-level keys so that
    shared defaults (e.g., sensitive/alpha/benchmarks) still apply.
    If 'profiles' is absent, treat the YAML as a single flat config (back-compatible).

    Args:
        path: Path to config file. If None and no text/obj provided, attempts discovery.
        text: YAML config as string (alternative to path)
        obj: Config as dict (alternative to path)
        profile: Profile name to use (if config has profiles)

    Returns:
        PipelineConfig instance

    Raises:
        ValueError: If multiple sources provided or config file not found
        ConfigValidationError: If config validation fails
    """
    provided = sum(x is not None for x in (path, text, obj))
    if provided > 1:
        raise ValueError("Provide exactly one of: path=, text=, or obj=.")

    # If no explicit source provided, attempt file discovery
    if provided == 0:
        discovered_path = find_config_file(explicit_path=path, profile=profile)
        if discovered_path is None:
            raise ValueError(
                "No config source provided and no config file found.\n"
                "Provide one of: path=, text=, or obj=, or place config at:\n"
                "  - .fairpipe/config.yml (project-local)\n"
                "  - ~/.fairpipe/config.yml (user-global)"
            )
        path = str(discovered_path)

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
    elif text is not None:
        root = yaml.safe_load(text) or {}
    else:
        root = obj or {}

    if not isinstance(root, dict):
        raise TypeError("Config must parse to a mapping/dict.")

    # Legacy single-config path (no profiles)
    if "profiles" not in root:
        return _as_cfg(root)

    profiles = root.get("profiles") or {}

    # Separate base/top-level defaults from profiles
    base = dict(root)
    base.pop("profiles", None)

    # Choose profile
    chosen = profile or os.getenv("FPDT_PROFILE") or _auto_choose_profile(profiles)

    if not chosen:
        avail = ", ".join(sorted(profiles.keys()))
        raise ValueError(
            "This config defines profiles, but no profile was selected.\n"
            "Pass profile in code/CLI (e.g., --profile pipeline) or set FPDT_PROFILE.\n"
            f"Available profiles: {avail}"
        )

    if chosen not in profiles:
        avail = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown profile '{chosen}'. Available: {avail}")

    # Merge selected profile over top-level defaults
    flat = _shallow_merge(base, profiles[chosen] or {})
    return _as_cfg(flat)
