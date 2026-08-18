"""Counterfactual fairness probe utilities."""

from .counterfactual import (
    CounterfactualPrompt,
    ResponseFeatures,
    divergence_by_dimension,
    extract_response_features,
    generate_counterfactual_prompts,
    pairwise_divergence,
)

__all__ = [
    "CounterfactualPrompt",
    "ResponseFeatures",
    "divergence_by_dimension",
    "extract_response_features",
    "generate_counterfactual_prompts",
    "pairwise_divergence",
]
