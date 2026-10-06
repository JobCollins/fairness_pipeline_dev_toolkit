"""Packaged LLM fixture helpers emit FutureWarning (Wave 4 / BL-030)."""

from __future__ import annotations

import warnings

import pytest


def test_default_recorded_toxicity_config_deprecated() -> None:
    from fairpipe.llm_evals import default_recorded_toxicity_config

    with pytest.warns(FutureWarning, match="deprecated"):
        cfg = default_recorded_toxicity_config()
    assert "toxicity" in cfg.evaluators[0]


def test_humanitarian_helpers_deprecated() -> None:
    from fairpipe.llm_evals import (
        humanitarian_contrast_config,
        humanitarian_divergence_config,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        humanitarian_divergence_config()
        humanitarian_contrast_config()
    messages = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert any("humanitarian_divergence_config" in m for m in messages)
    assert any("humanitarian_contrast_config" in m for m in messages)
