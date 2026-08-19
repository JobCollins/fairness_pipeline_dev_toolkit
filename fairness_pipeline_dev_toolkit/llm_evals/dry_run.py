from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError

# Approximate USD per 1K tokens (input, output) for dry-run estimates only.
MODEL_COST_PER_1K: Dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "local": (0.0, 0.0),
}

DEFAULT_INPUT_TOKENS_PER_PROMPT = 120
DEFAULT_OUTPUT_TOKENS_PER_RESPONSE = 80


@dataclass
class DryRunEstimate:
    request_count: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    breakdown: Dict[str, int]

    def to_markdown(self) -> str:
        lines = [
            "# LLM Eval Dry Run",
            "",
            f"- **Estimated requests**: {self.request_count}",
            f"- **Estimated input tokens**: {self.estimated_input_tokens:,}",
            f"- **Estimated output tokens**: {self.estimated_output_tokens:,}",
            f"- **Estimated cost (USD)**: ${self.estimated_cost_usd:.4f}",
            "",
            "## Request breakdown",
            "",
        ]
        for key, count in sorted(self.breakdown.items()):
            lines.append(f"- `{key}`: {count}")
        lines.append("")
        lines.append(
            "> Cost is approximate based on model list pricing and average token heuristics. "
            "No live provider calls are made during a dry run."
        )
        return "\n".join(lines)


def _model_pricing(model: str) -> tuple[float, float]:
    if model in MODEL_COST_PER_1K:
        return MODEL_COST_PER_1K[model]
    if model.startswith("gpt-"):
        return MODEL_COST_PER_1K["gpt-4o-mini"]
    if model.startswith("claude-"):
        return MODEL_COST_PER_1K["claude-3-haiku-20240307"]
    return (0.0, 0.0)


def estimate_counterfactual_requests(
    dimensions: Dict[str, List[str]],
    *,
    n_templates: int = 1,
) -> tuple[int, Dict[str, int]]:
    if n_templates < 1:
        raise ConfigValidationError("counterfactual.template must contain at least one template.")
    breakdown: Dict[str, int] = {}
    total = 0
    for dimension, values in dimensions.items():
        count = len(values) * n_templates
        if len(values) >= 2:
            breakdown[f"counterfactual:{dimension}"] = count
            total += count
    return total, breakdown


def estimate_dry_run(
    *,
    provider: str,
    model: str,
    evaluators: List[str],
    counterfactual_dimensions: Optional[Dict[str, List[str]]] = None,
    n_templates: int = 1,
    bbq_item_count: Optional[int] = None,
    input_tokens_per_prompt: int = DEFAULT_INPUT_TOKENS_PER_PROMPT,
    output_tokens_per_response: int = DEFAULT_OUTPUT_TOKENS_PER_RESPONSE,
) -> DryRunEstimate:
    breakdown: Dict[str, int] = {}
    request_count = 0

    if "counterfactual_fairness_divergence" in evaluators:
        if not counterfactual_dimensions:
            raise ConfigValidationError(
                "counterfactual.dimensions is required when running "
                "counterfactual_fairness_divergence."
            )
        cf_count, cf_breakdown = estimate_counterfactual_requests(
            counterfactual_dimensions, n_templates=n_templates
        )
        request_count += cf_count
        breakdown.update(cf_breakdown)

    prompt_group_evals = ("refusal_rate_disparity", "toxicity_sentiment_disparity")
    for evaluator in prompt_group_evals:
        if evaluator not in evaluators:
            continue
        if not counterfactual_dimensions:
            raise ConfigValidationError(
                f"counterfactual.dimensions is required when running {evaluator}."
            )
        n, br = estimate_counterfactual_requests(counterfactual_dimensions, n_templates=n_templates)
        request_count += n
        breakdown[evaluator] = n

    if "stereotype_association_score" in evaluators:
        n_bbq = bbq_item_count if bbq_item_count is not None else 12
        request_count += n_bbq
        breakdown["stereotype_association_score"] = n_bbq

    input_tokens = request_count * input_tokens_per_prompt
    output_tokens = request_count * output_tokens_per_response
    in_rate, out_rate = _model_pricing(model)
    cost = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
    if provider == "local":
        cost = 0.0

    return DryRunEstimate(
        request_count=request_count,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=cost,
        breakdown=breakdown,
    )
