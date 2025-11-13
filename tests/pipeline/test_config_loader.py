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
