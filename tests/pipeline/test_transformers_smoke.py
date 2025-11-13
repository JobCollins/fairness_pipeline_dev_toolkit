import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
)


def test_pipeline_build_and_apply_smoke():
    cfg = load_config("pipeline.config.yml")
    df = pd.read_csv("dev_sample.csv")
    pipe = build_pipeline(cfg)
    Xt, artifacts = apply_pipeline(pipe, df)
    assert Xt.shape[0] == df.shape[0]
    # if reweight present, we should have sample_weight
    if artifacts and "sample_weight" in artifacts:
        w = artifacts["sample_weight"]
        assert w is not None and len(w) == len(df)


def test_pipeline_builder_returns_sample_weights_from_yaml():
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

    pipe = build_pipeline(cfg)
    Xt, artifacts = apply_pipeline(pipe, df)

    assert isinstance(Xt, pd.DataFrame)
    assert artifacts is not None
    assert "sample_weight" in artifacts
    assert len(artifacts["sample_weight"]) == len(df)
