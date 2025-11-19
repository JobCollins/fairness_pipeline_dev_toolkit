"""
Enhanced smoke tests for pipeline transformers.
"""

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
)


def test_pipeline_build_and_apply_smoke(tmp_path):
    """Test pipeline build and apply with enhanced assertions."""
    # Create sample config file
    config_text = """
sensitive: ["group"]
pipeline:
  - name: reweigh
    transformer: InstanceReweighting
    params: {}
"""
    config_path = tmp_path / "pipeline.config.yml"
    config_path.write_text(config_text, encoding="utf-8")

    # Create sample data file
    sample_data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "A", "B"] * 5,
            "feature1": np.random.randn(40),
            "feature2": np.random.randn(40),
            "target": np.random.randint(0, 2, 40),
        }
    )
    data_path = tmp_path / "dev_sample.csv"
    sample_data.to_csv(data_path, index=False)

    # Load config and data
    cfg = load_config(str(config_path))
    df = pd.read_csv(data_path)

    # Build pipeline
    pipe = build_pipeline(cfg)
    assert pipe is not None, "Pipeline should be built successfully"

    # Apply pipeline
    Xt, artifacts = apply_pipeline(pipe, df)

    # Enhanced assertions for transformed data
    assert Xt is not None, "Transformed data should not be None"
    assert isinstance(Xt, pd.DataFrame), "Transformed data should be a DataFrame"
    assert Xt.shape[0] == df.shape[0], "Number of rows should be preserved"
    assert Xt.shape[1] > 0, "Transformed data should have columns"

    # Verify data integrity
    assert not Xt.empty, "Transformed data should not be empty"
    assert Xt.index.equals(df.index), "Index should be preserved"

    # Enhanced assertions for artifacts
    assert artifacts is not None, "Artifacts should not be None"
    assert isinstance(artifacts, dict), "Artifacts should be a dictionary"

    # If reweight present, we should have sample_weight
    if "sample_weight" in artifacts:
        w = artifacts["sample_weight"]
        assert w is not None, "Sample weights should not be None"
        assert len(w) == len(df), "Sample weights length should match data length"
        assert isinstance(w, np.ndarray), "Sample weights should be numpy array"
        assert np.all(w > 0), "All sample weights should be positive"
        assert np.isfinite(w).all(), "All sample weights should be finite"


def test_pipeline_builder_returns_sample_weights_from_yaml():
    """Test that pipeline builder correctly returns sample weights with enhanced validation."""
    cfg_text = """
    sensitive: ["s"]
    pipeline:
      - name: reweight
        transformer: InstanceReweighting
    """
    cfg = load_config(text=cfg_text)
    df = pd.DataFrame(
        {
            "s": ["A", "A", "B", "B", "C", "C"],
            "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    # Build and apply pipeline
    pipe = build_pipeline(cfg)
    assert pipe is not None, "Pipeline should be built successfully"

    Xt, artifacts = apply_pipeline(pipe, df)

    # Enhanced assertions for transformed data
    assert isinstance(Xt, pd.DataFrame), "Transformed data should be a DataFrame"
    assert Xt.shape[0] == df.shape[0], "Number of rows should be preserved"
    assert "s" in Xt.columns, "Sensitive column should be preserved"
    assert "feature" in Xt.columns, "Feature column should be preserved"

    # Enhanced assertions for artifacts
    assert artifacts is not None, "Artifacts should not be None"
    assert isinstance(artifacts, dict), "Artifacts should be a dictionary"
    assert "sample_weight" in artifacts, "Sample weights should be in artifacts"

    # Validate sample weights
    weights = artifacts["sample_weight"]
    assert len(weights) == len(df), "Sample weights length should match data length"
    assert isinstance(weights, np.ndarray), "Sample weights should be numpy array"
    assert np.all(weights > 0), "All sample weights should be positive"
    assert np.isfinite(weights).all(), "All sample weights should be finite"

    # Verify weights are normalized (mean should be close to 1.0)
    assert abs(weights.mean() - 1.0) < 0.1, "Sample weights should be normalized (mean ≈ 1.0)"
