from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import yaml

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError

VALID_PROVIDERS = frozenset({"openai", "anthropic", "local"})
VALID_EVALUATORS = frozenset(
    {
        "counterfactual_fairness_divergence",
        "refusal_rate_disparity",
        "toxicity_sentiment_disparity",
        "stereotype_association_score",
    }
)
CREDENTIAL_FIELD_NAMES = frozenset(
    {
        "api_key",
        "api-key",
        "apikey",
        "token",
        "secret",
        "password",
        "anthropic_api_key",
        "openai_api_key",
    }
)


@dataclass
class CounterfactualConfig:
    template: str | List[str]
    dimensions: Dict[str, List[str]]
    defaults: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMEvalConfig:
    provider: str
    model: str
    evaluators: List[str]
    prompt_templates: Dict[str, List[str]] = field(default_factory=dict)
    counterfactual: Optional[CounterfactualConfig] = None
    cache_dir: Optional[str] = None
    max_requests_per_run: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    bbq_path: Optional[str] = None
    allow_small_samples: bool = False


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


def _reject_credential_fields(raw: Dict[str, Any], *, prefix: str = "") -> None:
    for key, value in raw.items():
        normalized = str(key).lower().replace("-", "_")
        if normalized in CREDENTIAL_FIELD_NAMES or "api_key" in normalized:
            field_path = f"{prefix}{key}" if prefix else str(key)
            raise ConfigValidationError(
                f"Credential field '{field_path}' is not allowed in YAML config. "
                "Set provider API keys via environment variables only.",
                field_name=field_path,
            )
        if isinstance(value, dict):
            _reject_credential_fields(value, prefix=f"{prefix}{key}.")


def _validate_prompt_templates(value: Any) -> Dict[str, List[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigValidationError("Config field 'prompt_templates' must be a mapping.")
    out: Dict[str, List[str]] = {}
    for name, templates in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigValidationError(
                "Config field 'prompt_templates' keys must be non-empty strings."
            )
        out[name.strip()] = _ensure_list_of_strings(templates, f"prompt_templates.{name}")
    return out


def _validate_dimensions(value: Any, field_name: str) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Config field '{field_name}' must be a mapping.")
    out: Dict[str, List[str]] = {}
    for name, values in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigValidationError(
                f"Config field '{field_name}' keys must be non-empty strings."
            )
        cleaned = _ensure_list_of_strings(values, f"{field_name}.{name}")
        if len(cleaned) < 2:
            raise ConfigValidationError(
                f"Config field '{field_name}.{name}' must list at least two swap values."
            )
        out[name.strip()] = cleaned
    return out


def _validate_counterfactual_block(value: Any) -> Optional[CounterfactualConfig]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigValidationError("Config field 'counterfactual' must be a mapping.")
    template = value.get("template")
    if isinstance(template, list):
        cleaned_template: str | List[str] = _ensure_list_of_strings(
            template, "counterfactual.template"
        )
    elif isinstance(template, str) and template.strip():
        cleaned_template = template.strip()
    else:
        raise ConfigValidationError(
            "Config field 'counterfactual.template' must be a non-empty string "
            "or a list of non-empty strings."
        )
    dimensions = _validate_dimensions(value.get("dimensions"), "counterfactual.dimensions")
    if not dimensions:
        raise ConfigValidationError("Config field 'counterfactual.dimensions' must not be empty.")
    defaults = value.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ConfigValidationError(
            "Config field 'counterfactual.defaults' must be a mapping if provided."
        )
    cleaned_defaults = {str(k): str(v) for k, v in defaults.items()}
    return CounterfactualConfig(
        template=cleaned_template,
        dimensions=dimensions,
        defaults=cleaned_defaults,
    )


def _validate_llm_eval_block(raw: Dict[str, Any]) -> LLMEvalConfig:
    _reject_credential_fields(raw)

    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ConfigValidationError("Config field 'provider' must be a non-empty string.")
    provider = provider.strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ConfigValidationError(
            f"Config field 'provider' must be one of {sorted(VALID_PROVIDERS)}; got {provider!r}."
        )

    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigValidationError("Config field 'model' must be a non-empty string.")
    model = model.strip()

    evaluators = _ensure_list_of_strings(raw.get("evaluators"), "evaluators")
    unknown = [e for e in evaluators if e not in VALID_EVALUATORS]
    if unknown:
        raise ConfigValidationError(
            f"Unknown evaluator(s): {unknown}. Valid evaluators: {sorted(VALID_EVALUATORS)}."
        )

    params = raw.get("params")
    if params is not None and not isinstance(params, dict):
        raise ConfigValidationError("Config field 'params' must be a mapping if provided.")

    cache_dir = raw.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise ConfigValidationError("Config field 'cache_dir' must be a string when provided.")

    max_requests = raw.get("max_requests_per_run")
    if max_requests is not None:
        try:
            max_requests = int(max_requests)
            if max_requests <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ConfigValidationError(
                "Config field 'max_requests_per_run' must be a positive integer when provided."
            ) from None

    counterfactual = _validate_counterfactual_block(raw.get("counterfactual"))
    needs_counterfactual = {
        "counterfactual_fairness_divergence",
        "refusal_rate_disparity",
        "toxicity_sentiment_disparity",
    }
    if needs_counterfactual.intersection(evaluators) and counterfactual is None:
        raise ConfigValidationError(
            "Config field 'counterfactual' is required when "
            f"{sorted(needs_counterfactual)} is listed in evaluators."
        )

    bbq_path = raw.get("bbq_path")
    if bbq_path is not None and not isinstance(bbq_path, str):
        raise ConfigValidationError("Config field 'bbq_path' must be a string when provided.")

    allow_small_samples = raw.get("allow_small_samples", False)
    if not isinstance(allow_small_samples, bool):
        raise ConfigValidationError(
            "Config field 'allow_small_samples' must be a boolean when provided."
        )

    return LLMEvalConfig(
        provider=provider,
        model=model,
        evaluators=evaluators,
        prompt_templates=_validate_prompt_templates(raw.get("prompt_templates")),
        counterfactual=counterfactual,
        cache_dir=cache_dir,
        max_requests_per_run=max_requests,
        params=dict(params or {}),
        bbq_path=bbq_path,
        allow_small_samples=allow_small_samples,
    )


def _extract_llm_eval_root(root: Dict[str, Any]) -> Dict[str, Any]:
    if "llm_eval" in root:
        block = root["llm_eval"]
        if not isinstance(block, dict):
            raise ConfigValidationError("Config field 'llm_eval' must be a mapping.")
        return block
    return root


def load_llm_eval_config(
    path: Optional[str] = None,
    *,
    text: Optional[str] = None,
    obj: Optional[Dict[str, Any]] = None,
) -> LLMEvalConfig:
    """
    Load and validate an ``llm_eval`` YAML block.

    Accepts either a top-level ``llm_eval:`` mapping or a standalone file whose
    root keys are the block fields (``provider``, ``model``, ``evaluators``, ...).
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

    block = _extract_llm_eval_root(root)
    return _validate_llm_eval_block(block)
