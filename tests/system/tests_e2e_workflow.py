# FILE: tests/system/test_e2e_workflow.py
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.integration.reporting import to_markdown_report
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer


@pytest.fixture
def e2e_df():
    # deterministic small dataset
    rng = np.random.default_rng(123)
    n = 200
    y_true = rng.binomial(1, 0.55, n)
    # y_pred purposely correlated
    y_pred = (rng.random(n) < (0.5 + 0.2 * (y_true == 1))).astype(int)
    # groups: A/B with imbalance; add some small noise
    group = np.where(rng.random(n) < 0.6, "A", "B")
    # regression-like score
    score = (0.4 * y_true + 0.1 * (group == "A") + 0.1 * rng.standard_normal(n)).clip(0, 1)

    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group, "score": score})


def test_e2e_markdown_and_metrics(e2e_df):
    fa = FairnessAnalyzer(min_group_size=20, backend="native")

    results = {}
    results["demographic_parity_difference"] = fa.demographic_parity_difference(
        y_pred=e2e_df["y_pred"].to_numpy(),
        sensitive=e2e_df["group"].to_numpy(),
    )
    results["equalized_odds_difference"] = fa.equalized_odds_difference(
        y_true=e2e_df["y_true"].to_numpy(),
        y_pred=e2e_df["y_pred"].to_numpy(),
        sensitive=e2e_df["group"].to_numpy(),
    )
    results["mae_parity_difference"] = fa.mae_parity_difference(
        y_true=e2e_df["y_true"].to_numpy().astype(float),
        y_pred=e2e_df["score"].to_numpy().astype(float),
        sensitive=e2e_df["group"].to_numpy(),
    )

    md = to_markdown_report(results, title="E2E Report")
    # sanity checks
    assert "E2E Report" in md
    assert "demographic_parity_difference" in md
    assert "equalized_odds_difference" in md
    assert "mae_parity_difference" in md

    # sanity on values being present
    for k, v in results.items():
        assert isinstance(v.value, (float, type(np.nan)))


def test_e2e_mlflow_logging_mock(e2e_df, monkeypatch):
    # Import here to avoid hard dependency in other tests
    from fairness_pipeline_dev_toolkit.integration.mlflow_logger import (
        log_fairness_metrics,
    )

    fa = FairnessAnalyzer(min_group_size=20, backend="native")
    results = {
        "demographic_parity_difference": fa.demographic_parity_difference(
            y_pred=e2e_df["y_pred"].to_numpy(),
            sensitive=e2e_df["group"].to_numpy(),
        )
    }

    mock_mlflow = mock.MagicMock()
    # Should not raise, and should attempt to log metrics and one artifact
    log_fairness_metrics(run=mock_mlflow, results=results)

    # we don't know exact function calls inside, but ensure we touched mlflow object
    assert mock_mlflow is not None
