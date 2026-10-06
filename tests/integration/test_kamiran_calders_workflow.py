"""Integration tests: KamiranCaldersReweighing through execute_workflow."""

from __future__ import annotations

import logging
import textwrap

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.pipeline.config import load_config

try:
    from fairness_pipeline_dev_toolkit.integration.orchestrator import execute_workflow

    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


def _biased_df(n: int = 240, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sensitive = np.array(["A"] * (n * 3 // 5) + ["B"] * (n - n * 3 // 5))
    rng.shuffle(sensitive)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    # Base-rate gap by group
    p = np.where(sensitive == "A", 0.7, 0.3)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"f0": f0, "f1": f1, "sensitive": sensitive, "y": y})


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_execute_workflow_kc_reductions_passes_train_length_weights(caplog):
    """KC under reductions: sample weights reach the mixer / model path."""
    df = _biased_df()
    cfg = load_config(
        text=textwrap.dedent(
            """
            sensitive: ["sensitive"]
            features: ["f0", "f1"]
            pipeline:
              - name: kc
                transformer: "KamiranCaldersReweighing"
                params: {}
            training:
              method: "reductions"
              target_column: "y"
              params:
                constraint: "demographic_parity"
                eps: 0.05
                T: 5
            fairness_metric: "demographic_parity_difference"
            validation_threshold: 0.5
            """
        )
    )
    # Spy via apply_pipeline result by patching build path is heavy; instead
    # assert workflow succeeds and artifacts include weight stats when present.
    # Direct check: re-run the pipeline step the orchestrator uses.
    from fairness_pipeline_dev_toolkit.integration.orchestrator import (
        _make_workflow_split,
        _strip_sensitive_columns,
    )
    from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
        apply_pipeline,
        build_pipeline,
    )

    split = _make_workflow_split(df, cfg, train_size=0.75, random_state=0)
    pipe = build_pipeline(cfg)
    X_train_pipe = split.X_train.copy()
    X_train_pipe["sensitive"] = df.loc[split.X_train.index, "sensitive"]
    pr = apply_pipeline(pipe, X_train_pipe, y=split.y_train, fit=True)
    assert pr.sample_weight is not None
    assert pr.sample_weight.shape[0] == len(split.X_train)
    assert pr.sample_weight.sum() == pytest.approx(float(len(split.X_train)))
    # Label must not appear in model features
    Xt = _strip_sensitive_columns(pr.data, cfg.sensitive)
    assert "y" not in Xt.columns
    assert list(Xt.columns) == ["f0", "f1"]

    result = execute_workflow(cfg, df, train_size=0.75, random_state=0, min_group_size=10)
    assert result.model is not None
    assert result.predictions is not None


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training dependencies not available")
def test_execute_workflow_kc_regularized_logs_ignore_warning(caplog):
    pytest.importorskip("torch")
    df = _biased_df(n=120)
    cfg = load_config(
        text=textwrap.dedent(
            """
            sensitive: ["sensitive"]
            features: ["f0", "f1"]
            pipeline:
              - name: kc
                transformer: "KamiranCaldersReweighing"
                params: {}
            training:
              method: "regularized"
              target_column: "y"
              params:
                eta: 0.5
                epochs: 2
                lr: 0.01
            fairness_metric: "demographic_parity_difference"
            validation_threshold: 0.5
            """
        )
    )
    with caplog.at_level(logging.WARNING):
        execute_workflow(cfg, df, train_size=0.75, random_state=1, min_group_size=5)
    assert any("regularized" in r.message and "sample weights" in r.message for r in caplog.records)
