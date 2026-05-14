"""Tests for PipelineResult and apply_pipeline return type."""

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    PipelineResult,
    apply_pipeline,
    build_pipeline,
)


def test_apply_pipeline_returns_pipeline_result():
    cfg = load_config(
        text="""
sensitive: ["g"]
pipeline:
  - name: rw
    transformer: InstanceReweighting
"""
    )
    df = pd.DataFrame({"g": ["A", "B"], "x": [1.0, 2.0]})
    pipe = build_pipeline(cfg)
    result = apply_pipeline(pipe, df)
    assert isinstance(result, PipelineResult)
    assert isinstance(result.data, pd.DataFrame)
    assert result.transformers_applied == ("rw",)
    assert result.metadata is not None
    assert "sample_weight" in result.metadata
    assert result.sample_weight is not None
    assert len(result.sample_weight) == 2


def test_tuple_unpack_emits_deprecation_warning():
    cfg = load_config(
        text="""
sensitive: ["g"]
pipeline:
  - name: rw
    transformer: InstanceReweighting
"""
    )
    df = pd.DataFrame({"g": ["A", "B"], "x": [1.0, 2.0]})
    pipe = build_pipeline(cfg)
    result = apply_pipeline(pipe, df)
    with pytest.warns(DeprecationWarning, match="Unpacking apply_pipeline"):
        Xt, meta = result
    assert isinstance(Xt, pd.DataFrame)
    assert meta is result.metadata


def test_no_instance_reweighting_metadata_none():
    cfg = load_config(
        text="""
sensitive: ["g"]
benchmarks:
  g:
    "A": 0.5
    "B": 0.5
pipeline:
  - name: r
    transformer: ReweighingTransformer
    params: {}
"""
    )
    df = pd.DataFrame({"g": ["A", "B", "A", "B"], "x": [1.0, 2.0, 3.0, 4.0]})
    pipe = build_pipeline(cfg)
    result = apply_pipeline(pipe, df)
    assert result.metadata is None
    assert result.sample_weight is None
