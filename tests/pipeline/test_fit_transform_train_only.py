"""BL-025: fit on train only; transform test without refit; batch-stable DIR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fairness_pipeline_dev_toolkit.pipeline.config import PipelineConfig, PipelineStep
from fairness_pipeline_dev_toolkit.pipeline.orchestration.engine import (
    apply_pipeline,
    build_pipeline,
)
from fairness_pipeline_dev_toolkit.pipeline.transformers.disparate_impact import (
    DisparateImpactRemover,
)
from fairness_pipeline_dev_toolkit.pipeline.transformers.reweighing import (
    ReweighingTransformer,
)


def _dir_frames():
    """Train pool [0, 39]; test values in [100, 139] so a refit would shift the pool."""
    rng = np.random.default_rng(0)
    n_per = 40
    train = pd.DataFrame(
        {
            "score": np.concatenate(
                [
                    rng.uniform(0, 39, size=n_per),
                    rng.uniform(0, 39, size=n_per),
                ]
            ),
            "group": ["A"] * n_per + ["B"] * n_per,
        }
    )
    test = pd.DataFrame(
        {
            "score": np.concatenate(
                [
                    rng.uniform(100, 139, size=n_per),
                    rng.uniform(100, 139, size=n_per),
                ]
            ),
            "group": ["A"] * n_per + ["B"] * n_per,
        }
    )
    return train, test


def test_apply_pipeline_fit_false_keeps_train_pool_quantiles():
    """Regression: test transform must not refit — pool stays train [~0, ~39]."""
    train, test = _dir_frames()
    cfg = PipelineConfig(
        sensitive=["group"],
        pipeline=[
            PipelineStep(
                name="dir",
                transformer="DisparateImpactRemover",
                params={
                    "sensitive": "group",
                    "features": ["score"],
                    "repair_level": 1.0,
                    "min_group_size": 20,
                },
            )
        ],
    )
    pipe = build_pipeline(cfg)
    apply_pipeline(pipe, train, fit=True)
    dir_step = pipe.named_steps["dir"]
    pool_after_train = dir_step.pool_values_["score"].copy()

    apply_pipeline(pipe, test, fit=False)
    pool_after_test = dir_step.pool_values_["score"]

    np.testing.assert_array_equal(pool_after_train, pool_after_test)
    assert pool_after_train.min() < 50
    assert pool_after_train.max() < 50
    # Contrast: a fresh fit on test would land in [100, 139]
    refit = DisparateImpactRemover(
        sensitive="group", features=["score"], repair_level=1.0, min_group_size=20
    )
    refit.fit(test)
    assert refit.pool_values_["score"].min() >= 100


def test_test_data_does_not_influence_fitted_parameters():
    """Leakage guard: mutating test after fit leaves pool_quantiles_ unchanged."""
    train, test = _dir_frames()
    remover = DisparateImpactRemover(
        sensitive="group", features=["score"], repair_level=1.0, min_group_size=20
    )
    remover.fit(train)
    pool_q = remover.pool_quantiles_["score"].copy()
    pool_v = remover.pool_values_["score"].copy()
    group_refs = {k: v.copy() for k, v in remover.group_values_["score"].items()}

    test_mut = test.copy()
    test_mut["score"] = test_mut["score"] + 1e6
    _ = remover.transform(test_mut)

    np.testing.assert_array_equal(remover.pool_quantiles_["score"], pool_q)
    np.testing.assert_array_equal(remover.pool_values_["score"], pool_v)
    for k, v in group_refs.items():
        np.testing.assert_array_equal(remover.group_values_["score"][k], v)


def test_dir_single_row_matches_batch_member():
    """Same row alone vs in batch → identical repaired value (fitted train CDF)."""
    train, test = _dir_frames()
    remover = DisparateImpactRemover(
        sensitive="group", features=["score"], repair_level=1.0, min_group_size=20
    )
    remover.fit(train)

    batch = remover.transform(test)
    alone = remover.transform(test.iloc[[0]])
    assert alone["score"].iloc[0] == batch["score"].iloc[0]


def test_dir_undersized_fit_group_left_unchanged():
    """Groups below min_group_size at fit are not repaired (defined small-batch policy)."""
    df = pd.DataFrame(
        {
            "score": [1.0, 2.0, 100.0, 101.0],
            "group": ["A", "A", "B", "B"],
        }
    )
    remover = DisparateImpactRemover(
        sensitive="group", features=["score"], repair_level=1.0, min_group_size=5
    )
    remover.fit(df)
    out = remover.transform(df)
    np.testing.assert_array_equal(out["score"].to_numpy(), df["score"].to_numpy())
    assert remover.repairable_groups_["score"] == set()


def test_reweighing_transform_does_not_refit_on_test():
    train = pd.DataFrame({"x": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
    test = pd.DataFrame({"x": [10, 20], "group": ["A", "B"]})
    t = ReweighingTransformer(sensitive=["group"])
    t.fit(train)
    w_train = t.sample_weight_.copy()
    t.transform(test)
    np.testing.assert_array_equal(t.sample_weight_, w_train)
    assert len(t.sample_weight_) == len(train)


def test_sklearn_pipeline_fit_then_transform_test():
    train, test = _dir_frames()
    pipe = Pipeline(
        [
            (
                "dir",
                DisparateImpactRemover(
                    sensitive="group",
                    features=["score"],
                    repair_level=1.0,
                    min_group_size=20,
                ),
            )
        ]
    )
    pipe.fit(train)
    pool = pipe.named_steps["dir"].pool_values_["score"].copy()
    _ = pipe.transform(test)
    np.testing.assert_array_equal(pipe.named_steps["dir"].pool_values_["score"], pool)
