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
