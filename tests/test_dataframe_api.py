"""Tests for the DataFrame-friendly API additions (D2).

Covers:
- pd.Series and list inputs to FairnessAnalyzer metric methods
- FairnessAnalyzer.from_dataframe() proxy
- execute_workflow accepting a DataFrame directly
- CLI validate accepting .parquet files
- CLI validate rejecting unsupported file extensions
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

# ---------------------------------------------------------------------------
# Shared small dataset (50 samples per group so min_group_size=30 is satisfied)
# ---------------------------------------------------------------------------


def _make_df(n_per_group: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    sensitive = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "sensitive": sensitive})


_DF = _make_df()


# ---------------------------------------------------------------------------
# Category 1: array-like inputs
# ---------------------------------------------------------------------------


def test_analyzer_accepts_series():
    analyzer = FairnessAnalyzer(min_group_size=30, backend="native")
    result = analyzer.demographic_parity_difference(
        y_pred=_DF["y_pred"],  # pd.Series
        sensitive=_DF["sensitive"],  # pd.Series
        with_ci=False,
        with_effect_size=False,
    )
    assert result.value is not None
    assert np.isfinite(result.value)


def test_analyzer_accepts_list():
    analyzer = FairnessAnalyzer(min_group_size=30, backend="native")
    result = analyzer.demographic_parity_difference(
        y_pred=list(_DF["y_pred"].values),
        sensitive=list(_DF["sensitive"].values),
        with_ci=False,
        with_effect_size=False,
    )
    assert result.value is not None
    assert np.isfinite(result.value)


# ---------------------------------------------------------------------------
# Category 2: FairnessAnalyzerDataFrameProxy / from_dataframe()
# ---------------------------------------------------------------------------


def test_from_dataframe_proxy_basic():
    proxy = FairnessAnalyzer.from_dataframe(
        _DF,
        y_pred_col="y_pred",
        sensitive_col="sensitive",
        y_true_col="y_true",
        min_group_size=30,
        backend="native",
    )
    result = proxy.demographic_parity_difference(with_ci=False, with_effect_size=False)
    assert result.value is not None
    assert np.isfinite(result.value)


def test_from_dataframe_with_ci():
    proxy = FairnessAnalyzer.from_dataframe(
        _DF,
        y_pred_col="y_pred",
        sensitive_col="sensitive",
        min_group_size=30,
        backend="native",
    )
    result = proxy.demographic_parity_difference(with_ci=True, ci_samples=200)
    assert result.ci is not None
    assert len(result.ci) == 2


def test_from_dataframe_missing_column():
    with pytest.raises(KeyError, match="not found in DataFrame"):
        FairnessAnalyzer.from_dataframe(
            _DF,
            y_pred_col="nonexistent_col",
            sensitive_col="sensitive",
        )


# ---------------------------------------------------------------------------
# Category 4: execute_workflow accepts DataFrame directly
# ---------------------------------------------------------------------------


def test_execute_workflow_accepts_dataframe(tmp_path):
    """execute_workflow already accepts pd.DataFrame — confirm it still works."""

    from fairness_pipeline_dev_toolkit.integration import execute_workflow
    from fairness_pipeline_dev_toolkit.pipeline import load_config

    # Build a minimal config with a training section
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
sensitive: ["sensitive"]
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
training:
  method: "reductions"
  target_column: "y_true"
  params:
    constraint: "demographic_parity"
    eps: 0.05
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.5
""",
        encoding="utf-8",
    )

    df = _make_df(n_per_group=60)
    cfg = load_config(str(config_path))
    result = execute_workflow(config=cfg, df=df, min_group_size=30, train_size=0.8)
    assert hasattr(result, "validation_result")
    assert isinstance(result.validation_result.passed, bool)


# ---------------------------------------------------------------------------
# Category 3: CLI validate accepts .parquet
# ---------------------------------------------------------------------------


def test_cli_validate_parquet(tmp_path):
    df = _make_df(n_per_group=50)
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fairness_pipeline_dev_toolkit.cli.main",
            "validate",
            "--csv",
            str(parquet_path),
            "--y-true",
            "y_true",
            "--y-pred",
            "y_pred",
            "--sensitive",
            "sensitive",
            "--min-group-size",
            "30",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_validate_unsupported_format(tmp_path):
    bad_path = tmp_path / "data.txt"
    bad_path.write_text("y_true,y_pred,sensitive\n1,1,A\n")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fairness_pipeline_dev_toolkit.cli.main",
            "validate",
            "--csv",
            str(bad_path),
            "--y-true",
            "y_true",
            "--y-pred",
            "y_pred",
            "--sensitive",
            "sensitive",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unsupported file format" in proc.stdout + proc.stderr
