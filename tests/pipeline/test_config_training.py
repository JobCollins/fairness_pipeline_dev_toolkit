"""
Tests for extended config schema with training section.
"""

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


def test_load_config_with_training_section(tmp_path):
    """Test loading config with training section."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "pipeline": [],
            "training": {
                "method": "reductions",
                "target_column": "y",
                "params": {"constraint": "demographic_parity", "eps": 0.01},
            },
            "fairness_metric": "demographic_parity_difference",
            "validation_threshold": 0.05,
        },
    )

    cfg = load_config(path=str(cfg_path))
    assert cfg.training is not None
    assert cfg.training.method == "reductions"
    assert cfg.training.target_column == "y"
    assert cfg.training.params["constraint"] == "demographic_parity"
    assert cfg.fairness_metric == "demographic_parity_difference"
    assert cfg.validation_threshold == 0.05


def test_load_config_training_validation_method(tmp_path):
    """Test that training method must be valid."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "training": {
                "method": "invalid_method",
                "target_column": "y",
            },
        },
    )

    with pytest.raises(ConfigValidationError, match="training.method"):
        load_config(path=str(cfg_path))


def test_load_config_training_requires_target_column(tmp_path):
    """Test that training requires target_column."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "training": {
                "method": "reductions",
            },
        },
    )

    with pytest.raises(ConfigValidationError, match="target_column"):
        load_config(path=str(cfg_path))


def test_load_config_training_all_methods(tmp_path):
    """Test all three training methods are accepted."""
    methods = ["reductions", "regularized", "lagrangian"]
    for method in methods:
        cfg_path = _write_yaml(
            tmp_path,
            {
                "sensitive": ["sensitive"],
                "training": {
                    "method": method,
                    "target_column": "y",
                },
            },
        )
        cfg = load_config(path=str(cfg_path))
        assert cfg.training.method == method


def test_load_config_validation_threshold_numeric(tmp_path):
    """Test that validation_threshold must be numeric."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "validation_threshold": "not_a_number",
        },
    )

    with pytest.raises(ConfigValidationError, match="validation_threshold"):
        load_config(path=str(cfg_path))


def test_load_config_fairness_metric_string(tmp_path):
    """Test that fairness_metric must be a string."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "fairness_metric": 123,  # Should be string
        },
    )

    with pytest.raises(ConfigValidationError, match="fairness_metric"):
        load_config(path=str(cfg_path))


def test_load_config_backward_compatible_no_training(tmp_path):
    """Test that configs without training section still work (backward compatibility)."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "pipeline": [
                {"name": "step", "transformer": "InstanceReweighting"},
            ],
        },
    )

    cfg = load_config(path=str(cfg_path))
    assert cfg.training is None
    assert cfg.fairness_metric is None
    assert cfg.validation_threshold is None
    assert len(cfg.pipeline) == 1


def test_config_with_profile_and_training(tmp_path):
    """Test config with profiles and training section."""
    cfg_path = _write_yaml(
        tmp_path,
        {
            "sensitive": ["sensitive"],
            "profiles": {
                "training": {
                    "training": {
                        "method": "reductions",
                        "target_column": "y",
                        "params": {"constraint": "demographic_parity"},
                    },
                    "fairness_metric": "demographic_parity_difference",
                    "validation_threshold": 0.05,
                }
            },
        },
    )

    cfg = load_config(path=str(cfg_path), profile="training")
    assert cfg.training is not None
    assert cfg.training.method == "reductions"
    assert cfg.fairness_metric == "demographic_parity_difference"
