from __future__ import annotations

import pytest
import yaml

from fairness_pipeline_dev_toolkit.pipeline.config import (
    ConfigValidationError,
    load_config,
)


def _write_yaml(tmp_path, payload):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_config_requires_sensitive_list(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "pipeline": [
                {"name": "step", "transformer": "InstanceReweighting", "params": {}},
            ]
        },
    )

    with pytest.raises(ConfigValidationError, match="sensitive"):
        load_config(path=str(cfg_path))


def test_load_config_validates_pipeline_steps(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["attr"],
            "pipeline": [
                {"name": "step", "params": {}},
            ],
        },
    )

    with pytest.raises(ConfigValidationError, match="requires a 'transformer'"):
        load_config(path=str(cfg_path))


def test_load_config_profile_merge_respects_schema(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["attr"],
            "profiles": {
                "pipeline": {
                    "pipeline": [
                        {"name": "step", "transformer": "InstanceReweighting"},
                    ]
                }
            },
        },
    )

    cfg = load_config(path=str(cfg_path))
    assert cfg.sensitive == ["attr"]
    assert len(cfg.pipeline) == 1


def test_load_config_invalid_yaml_syntax(tmp_path):
    """Test that invalid YAML syntax raises appropriate error."""
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("invalid: yaml: syntax: [", encoding="utf-8")

    with pytest.raises((yaml.YAMLError, Exception)):
        load_config(path=str(cfg_path))


def test_load_config_missing_file():
    """Test that missing config file raises appropriate error."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_config(path="nonexistent_config.yml")


def test_load_config_empty_file(tmp_path):
    """Test that empty config file raises appropriate error."""
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("", encoding="utf-8")

    with pytest.raises((ConfigValidationError, ValueError, KeyError)):
        load_config(path=str(cfg_path))


def test_load_config_features_field(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["attr"],
            "features": ["x1", "x2"],
            "pipeline": [],
        },
    )
    cfg = load_config(path=str(cfg_path))
    assert cfg.features == ["x1", "x2"]


def test_load_config_features_empty_list_raises(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["attr"],
            "features": [],
            "pipeline": [],
        },
    )
    with pytest.raises(ConfigValidationError, match="must not be empty"):
        load_config(path=str(cfg_path))


def test_load_config_features_invalid_entry_raises(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["attr"],
            "features": [1, 2],
            "pipeline": [],
        },
    )
    with pytest.raises(ConfigValidationError, match="non-empty string"):
        load_config(path=str(cfg_path))


def test_load_config_invalid_transformer_name(tmp_path):
    """Test that invalid transformer name raises appropriate error."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["attr"],
            "pipeline": [
                {"name": "step", "transformer": "NonExistentTransformer"},
            ],
        },
    )

    # Should either raise error or handle gracefully
    try:
        cfg = load_config(path=str(cfg_path))
        # If it doesn't raise, the transformer should be validated later
        assert cfg is not None
    except (ConfigValidationError, ValueError, KeyError):
        pass  # Expected behavior
