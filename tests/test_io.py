"""Tests for fairness_pipeline_dev_toolkit.io.load_data()."""

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.io import load_data

_SAMPLE = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_load_csv(tmp_path):
    path = tmp_path / "data.csv"
    _SAMPLE.to_csv(path, index=False)
    df = load_data(path)
    assert df.shape == _SAMPLE.shape
    assert list(df.columns) == list(_SAMPLE.columns)


def test_load_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    _SAMPLE.to_parquet(path, index=False)
    df = load_data(path)
    assert df.shape == _SAMPLE.shape
    assert list(df.columns) == list(_SAMPLE.columns)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nonexistent.csv")
