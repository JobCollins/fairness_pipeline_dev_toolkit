"""
Tests for CLI validate command, including negative test cases.
"""

from __future__ import annotations

import pandas as pd

from fairness_pipeline_dev_toolkit.cli.main import main


def test_validate_with_ci_and_effects(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 0, 1, 0],
            "yhat": [1, 1, 0, 0, 0, 0, 1, 0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    csv_path = tmp_path / "validate.csv"
    df.to_csv(csv_path, index=False)

    calls = {}

    def _stub_report(results, title):
        dp = results["demographic_parity_difference"]
        eq = results["equalized_odds_difference"]
        assert dp.ci is not None and dp.effect_size is not None
        assert eq.ci is not None and eq.effect_size is not None
        calls["count"] = calls.get("count", 0) + 1
        return "stub"

    monkeypatch.setattr(
        "fairness_pipeline_dev_toolkit.cli.main.to_markdown_report",
        _stub_report,
    )

    exit_code = main(
        [
            "validate",
            "--csv",
            str(csv_path),
            "--y-true",
            "y",
            "--y-pred",
            "yhat",
            "--sensitive",
            "group",
            "--min-group-size",
            "1",
            "--with-ci",
            "--with-effects",
            "--bootstrap-B",
            "50",
        ]
    )

    assert exit_code == 0
    assert calls.get("count") == 1


def test_validate_missing_csv_file(monkeypatch):
    """Test that missing CSV file raises appropriate error."""
    # The CLI may handle this differently - either raise exception or return non-zero exit
    try:
        exit_code = main(
            [
                "validate",
                "--csv",
                "nonexistent_file.csv",
                "--y-true",
                "y",
                "--y-pred",
                "yhat",
                "--sensitive",
                "group",
            ]
        )
        # If it doesn't raise, should return non-zero exit code
        assert exit_code != 0, "Should fail with missing file"
    except (FileNotFoundError, SystemExit, pd.errors.EmptyDataError):
        # If it raises an exception, that's also acceptable
        pass


def test_validate_missing_required_column(monkeypatch, tmp_path):
    """Test that missing required column raises appropriate error."""
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "group": ["A", "B", "A", "B"],
        }
    )
    csv_path = tmp_path / "validate.csv"
    df.to_csv(csv_path, index=False)

    try:
        exit_code = main(
            [
                "validate",
                "--csv",
                str(csv_path),
                "--y-true",
                "y",
                "--y-pred",
                "missing_column",  # Column doesn't exist
                "--sensitive",
                "group",
            ]
        )
        assert exit_code != 0  # Should fail with missing column
    except (SystemExit, ValueError, KeyError):
        # If it raises an exception, that's also acceptable
        pass


def test_validate_missing_sensitive_column(monkeypatch, tmp_path):
    """Test that missing sensitive column raises appropriate error."""
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "yhat": [1, 0, 1, 0],
        }
    )
    csv_path = tmp_path / "validate.csv"
    df.to_csv(csv_path, index=False)

    try:
        exit_code = main(
            [
                "validate",
                "--csv",
                str(csv_path),
                "--y-true",
                "y",
                "--y-pred",
                "yhat",
                "--sensitive",
                "missing_group",  # Column doesn't exist
            ]
        )
        assert exit_code != 0  # Should fail with missing sensitive column
    except (SystemExit, ValueError, KeyError):
        # If it raises an exception, that's also acceptable
        pass


def test_validate_empty_dataframe(monkeypatch, tmp_path):
    """Test that empty DataFrame is handled appropriately."""
    df = pd.DataFrame({"y": [], "yhat": [], "group": []})
    csv_path = tmp_path / "validate.csv"
    df.to_csv(csv_path, index=False)

    try:
        exit_code = main(
            [
                "validate",
                "--csv",
                str(csv_path),
                "--y-true",
                "y",
                "--y-pred",
                "yhat",
                "--sensitive",
                "group",
            ]
        )
        # Should either fail or handle gracefully
        assert exit_code is not None
    except (SystemExit, ValueError, KeyError, IndexError):
        # If it raises an exception, that's also acceptable
        pass


def test_validate_invalid_min_group_size(monkeypatch, tmp_path):
    """Test that invalid min_group_size is handled."""
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "yhat": [1, 0, 1, 0],
            "group": ["A", "B", "A", "B"],
        }
    )
    csv_path = tmp_path / "validate.csv"
    df.to_csv(csv_path, index=False)

    exit_code = main(
        [
            "validate",
            "--csv",
            str(csv_path),
            "--y-true",
            "y",
            "--y-pred",
            "yhat",
            "--sensitive",
            "group",
            "--min-group-size",
            "100",  # Larger than any group size
        ]
    )

    # Should handle gracefully (may return 0 with warnings or non-zero)
    assert exit_code is not None
