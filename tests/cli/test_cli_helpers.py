"""
Tests for CLI helper functions: _parse_sensitive, _normalize_sensitive_arg, _write_artifact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.cli.main import (
    _normalize_sensitive_arg,
    _parse_sensitive,
    _write_artifact,
)


class TestParseSensitive:
    """Tests for _parse_sensitive() helper function."""

    def test_parse_sensitive_single_column(self):
        """Test parsing a single sensitive column."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "sensitive": ["A", "B", "A"]})
        result = _parse_sensitive(df, ["sensitive"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["sensitive"]
        assert len(result) == 3

    def test_parse_sensitive_multiple_columns(self):
        """Test parsing multiple sensitive columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "race": ["A", "B", "A"],
                "gender": ["M", "F", "M"],
            }
        )
        result = _parse_sensitive(df, ["race", "gender"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["race", "gender"]
        assert len(result) == 3

    def test_parse_sensitive_empty_list(self):
        """Test that empty list raises SystemExit."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(SystemExit, match="No --sensitive columns provided"):
            _parse_sensitive(df, [])

    def test_parse_sensitive_missing_column(self):
        """Test that missing column raises SystemExit."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(SystemExit, match="Sensitive column not found"):
            _parse_sensitive(df, ["nonexistent"])

    def test_parse_sensitive_multiple_missing_columns(self):
        """Test that multiple missing columns are reported."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(SystemExit, match="Sensitive column not found"):
            _parse_sensitive(df, ["nonexistent1", "nonexistent2"])


class TestNormalizeSensitiveArg:
    """Tests for _normalize_sensitive_arg() helper function."""

    def test_normalize_string_single(self):
        """Test normalizing a single column string."""
        result = _normalize_sensitive_arg("race")
        assert result == ["race"]

    def test_normalize_string_multiple_comma_separated(self):
        """Test normalizing comma-separated string."""
        result = _normalize_sensitive_arg("race,gender")
        assert result == ["race", "gender"]

    def test_normalize_string_with_spaces(self):
        """Test normalizing string with spaces around commas."""
        result = _normalize_sensitive_arg("race , gender , age")
        assert result == ["race", "gender", "age"]

    def test_normalize_list_single(self):
        """Test normalizing a list with single item."""
        result = _normalize_sensitive_arg(["race"])
        assert result == ["race"]

    def test_normalize_list_multiple(self):
        """Test normalizing a list with multiple items."""
        result = _normalize_sensitive_arg(["race", "gender"])
        assert result == ["race", "gender"]

    def test_normalize_list_with_comma_separated_strings(self):
        """Test normalizing a list where items contain commas."""
        result = _normalize_sensitive_arg(["race,gender", "age"])
        assert result == ["race", "gender", "age"]

    def test_normalize_list_with_spaces(self):
        """Test normalizing a list with spaces."""
        result = _normalize_sensitive_arg(["race , gender", " age "])
        assert result == ["race", "gender", "age"]

    def test_normalize_tuple(self):
        """Test normalizing a tuple."""
        result = _normalize_sensitive_arg(("race", "gender"))
        assert result == ["race", "gender"]

    def test_normalize_empty_string(self):
        """Test normalizing empty string returns empty list."""
        result = _normalize_sensitive_arg("")
        assert result == []

    def test_normalize_string_with_empty_parts(self):
        """Test normalizing string with empty parts (extra commas)."""
        result = _normalize_sensitive_arg("race,,gender,")
        assert result == ["race", "gender"]


class TestWriteArtifact:
    """Tests for _write_artifact() helper function."""

    def test_write_artifact_creates_file(self, tmp_path):
        """Test that _write_artifact creates a file."""
        file_path = tmp_path / "test.txt"
        content = "test content"
        _write_artifact(str(file_path), content)
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_write_artifact_creates_directories(self, tmp_path):
        """Test that _write_artifact creates parent directories."""
        file_path = tmp_path / "subdir" / "nested" / "test.txt"
        content = "test content"
        _write_artifact(str(file_path), content)
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_write_artifact_with_none_path(self, tmp_path):
        """Test that _write_artifact does nothing when path is None."""
        _write_artifact(None, "test content")
        # Should not raise and should not create any file

    def test_write_artifact_with_empty_string_path(self, tmp_path):
        """Test that _write_artifact handles empty string path."""
        # Empty string is falsy, so it should be treated like None
        _write_artifact("", "test content")
        # Should not raise

    def test_write_artifact_append_mode(self, tmp_path):
        """Test that _write_artifact works with append mode."""
        file_path = tmp_path / "test.txt"
        _write_artifact(str(file_path), "first line\n", mode="w")
        _write_artifact(str(file_path), "second line\n", mode="a")
        content = file_path.read_text()
        assert "first line" in content
        assert "second line" in content

    def test_write_artifact_unicode_content(self, tmp_path):
        """Test that _write_artifact handles unicode content."""
        file_path = tmp_path / "test.txt"
        content = "测试内容 🎉"
        _write_artifact(str(file_path), content)
        assert file_path.read_text() == content

    def test_write_artifact_overwrites_existing(self, tmp_path):
        """Test that _write_artifact overwrites existing file."""
        file_path = tmp_path / "test.txt"
        _write_artifact(str(file_path), "original")
        _write_artifact(str(file_path), "updated")
        assert file_path.read_text() == "updated"
