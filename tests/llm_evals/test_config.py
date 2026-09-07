"""Tests for llm_eval YAML config loading and validation."""

from __future__ import annotations

import pytest
import yaml

from fairness_pipeline_dev_toolkit.exceptions import ConfigValidationError
from fairness_pipeline_dev_toolkit.llm_evals.config import load_llm_eval_config


def _yaml(payload):
    return yaml.safe_dump(payload)


def test_load_valid_llm_eval_block():
    cfg = load_llm_eval_config(
        text=_yaml(
            {
                "llm_eval": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "evaluators": ["counterfactual_fairness_divergence"],
                    "counterfactual": {
                        "template": "Describe a {role} named {name}.",
                        "dimensions": {
                            "role": ["nurse", "doctor"],
                        },
                        "defaults": {"name": "Alex"},
                    },
                    "params": {"temperature": 0.0},
                }
            }
        )
    )

    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.evaluators == ["counterfactual_fairness_divergence"]
    assert cfg.counterfactual is not None
    assert cfg.counterfactual.template.startswith("Describe")
    assert cfg.params == {"temperature": 0.0}


def test_load_valid_standalone_root_block():
    cfg = load_llm_eval_config(
        obj={
            "provider": "local",
            "model": "stub-model",
            "evaluators": ["stereotype_association_score"],
        }
    )
    assert cfg.provider == "local"
    assert cfg.model == "stub-model"


def test_rejects_missing_model():
    with pytest.raises(ConfigValidationError, match="model"):
        load_llm_eval_config(
            obj={
                "provider": "openai",
                "evaluators": ["counterfactual_fairness_divergence"],
            }
        )


def test_rejects_api_key_in_yaml():
    with pytest.raises(ConfigValidationError, match="Credential field"):
        load_llm_eval_config(
            obj={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "evaluators": ["counterfactual_fairness_divergence"],
                "api_key": "sk-secret",
            }
        )


def test_load_list_of_counterfactual_templates():
    cfg = load_llm_eval_config(
        obj={
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "evaluators": ["counterfactual_fairness_divergence"],
            "counterfactual": {
                "template": [
                    "Recommend {name}, a {gender} engineer.",
                    "Assess {name}, a {gender} engineer.",
                ],
                "dimensions": {"gender": ["woman", "man"]},
                "defaults": {"name": "Alex"},
            },
        }
    )
    assert cfg.counterfactual is not None
    assert isinstance(cfg.counterfactual.template, list)
    assert len(cfg.counterfactual.template) == 2
    assert cfg.counterfactual.name_pools == {}


def test_rejects_unknown_evaluator():
    with pytest.raises(ConfigValidationError, match="Unknown evaluator"):
        load_llm_eval_config(
            obj={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "evaluators": ["hallucination_score"],
            }
        )


def _name_pool_base(**counterfactual_extra):
    block = {
        "template": [
            "Recommend {gender} for the role.",
            "Assess {gender} for the role.",
        ],
        "dimensions": {"gender": ["woman", "man"]},
        "defaults": {},
    }
    block.update(counterfactual_extra)
    return {
        "provider": "local",
        "model": "stub-model",
        "evaluators": ["counterfactual_fairness_divergence"],
        "counterfactual": block,
    }


def test_load_valid_name_pools():
    cfg = load_llm_eval_config(
        obj=_name_pool_base(
            name_pools={
                "gender": {
                    "woman": ["Aisha", "Fatima"],
                    "man": ["Omar", "Ahmed"],
                }
            }
        )
    )
    assert cfg.counterfactual is not None
    assert cfg.counterfactual.name_pools == {
        "gender": {
            "woman": ["Aisha", "Fatima"],
            "man": ["Omar", "Ahmed"],
        }
    }


def test_rejects_name_pools_wrong_length():
    with pytest.raises(ConfigValidationError, match="exactly 2 values"):
        load_llm_eval_config(
            obj=_name_pool_base(
                name_pools={"gender": {"woman": ["Aisha"], "man": ["Omar", "Ahmed"]}}
            )
        )


def test_rejects_name_pools_unknown_group():
    with pytest.raises(ConfigValidationError, match="not one of the groups"):
        load_llm_eval_config(
            obj=_name_pool_base(
                name_pools={
                    "gender": {
                        "woman": ["Aisha", "Fatima"],
                        "nonbinary": ["Sam", "Alex"],
                    }
                }
            )
        )


def test_rejects_name_pools_unknown_dimension():
    with pytest.raises(ConfigValidationError, match="not present in counterfactual.dimensions"):
        load_llm_eval_config(
            obj=_name_pool_base(
                name_pools={
                    "ethnicity": {
                        "woman": ["Aisha", "Fatima"],
                    }
                }
            )
        )


def test_rejects_name_pools_wrong_type():
    with pytest.raises(ConfigValidationError, match="must be a mapping if provided"):
        load_llm_eval_config(obj=_name_pool_base(name_pools=["Aisha", "Omar"]))


def test_rejects_name_pools_group_values_wrong_type():
    with pytest.raises(ConfigValidationError, match="must be a mapping from group label"):
        load_llm_eval_config(obj=_name_pool_base(name_pools={"gender": ["Aisha", "Omar"]}))
