"""EOD undefined gating through CLI validate and assert_fairness (BL-019 EO half)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.integration.pytest_plugin import assert_fairness
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer


def _incomplete_eo_frame() -> pd.DataFrame:
    """Group B has no positives → EOD undefined."""
    sensitive = np.repeat(["a", "b"], 40)
    y_true = np.tile([1.0, 0.0], 40)
    y_true[40:] = 0.0
    y_pred = np.tile([1.0, 0.0, 0.0, 1.0], 20)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive})


def test_assert_fairness_reports_undefined_on_nan():
    with pytest.raises(AssertionError, match="undefined"):
        assert_fairness(float("nan"), threshold=0.1, context="eod")


def test_analyzer_eod_undefined_value_and_ci_note():
    df = _incomplete_eo_frame()
    fa = FairnessAnalyzer(min_group_size=10, backend="native")
    res = fa.equalized_odds_difference(df["y_true"], df["y_pred"], df["group"])
    assert math.isnan(res.value)
    assert res.ci is None
    assert res.ci_note is not None and res.ci_note.startswith("undefined:empty_label_stratum")
    assert res.p_value is None


def test_cli_validate_exits_undefined_on_incomplete_eod(tmp_path: Path):
    from fairness_pipeline_dev_toolkit.cli.main import main

    csv_path = tmp_path / "eod.csv"
    _incomplete_eo_frame().to_csv(csv_path, index=False)

    code = main(
        [
            "validate",
            "--csv",
            str(csv_path),
            "--y-true",
            "y_true",
            "--y-pred",
            "y_pred",
            "--sensitive",
            "group",
            "--min-group-size",
            "10",
            "--metric",
            "equalized_odds_difference",
            "--threshold",
            "0.1",
            "--quiet",
        ]
    )
    assert code == 4
