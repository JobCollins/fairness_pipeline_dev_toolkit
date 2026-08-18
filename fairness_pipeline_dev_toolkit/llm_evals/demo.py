"""Deterministic local responders for demos, notebooks, and tests."""

from __future__ import annotations

from typing import Any, Dict, Optional


def biased_hiring_responder(prompt: str, *, params: Optional[Dict[str, Any]] = None) -> str:
    """Simulate disparate LLM hiring recommendations across gender-coded prompts."""
    text = prompt.lower()
    if "woman" in text:
        return "Excellent candidate with outstanding leadership potential; strongly recommend."
    if "man" in text:
        return "I cannot recommend this candidate for leadership roles due to significant concerns."
    return "Neutral assessment with mixed indicators."


def build_local_demo_client(model: str):
    from .client import LocalLLMClient

    return LocalLLMClient(model, responder=biased_hiring_responder)
