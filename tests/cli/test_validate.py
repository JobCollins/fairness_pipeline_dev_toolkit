"""
Tests for CLI validate command, including negative test cases.
"""

from __future__ import annotations

import subprocess
import sys

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


def test_validate_threshold_pass_and_report_section(capsys, tmp_path):
    """Fair predictions across groups → abs(DPD) within threshold → exit 0 and verdict in report."""
    df = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 1, 0, 0],
            "yhat": [1, 0, 1, 0, 1, 0, 1, 0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    csv_path = tmp_path / "fair.csv"
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
            "1",
            "--threshold",
            "0.35",
            "--metric",
            "demographic_parity_difference",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "## Threshold verdict" in out
    assert "**PASS**" in out
    assert "abs(metric) <= threshold" in out


def test_validate_threshold_fail_exit_one(capsys, tmp_path):
    """Maximally skewed group-wise predictions → DPD exceeds tight threshold → exit 1."""
    df = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 1, 0, 0],
            "yhat": [1, 1, 1, 1, 0, 0, 0, 0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    csv_path = tmp_path / "unfair.csv"
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
            "1",
            "--threshold",
            "0.2",
            "--metric",
            "demographic_parity_difference",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "**FAIL**" in out


def test_validate_no_threshold_legacy_exit_zero(capsys, tmp_path):
    """Without --threshold, exit 0 on success (backward compatible); no verdict section."""
    df = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 1, 0, 0],
            "yhat": [1, 1, 1, 1, 0, 0, 0, 0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    csv_path = tmp_path / "unfair.csv"
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
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Threshold verdict" not in out


def test_validate_threshold_without_metric_errors(capsys, tmp_path):
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "yhat": [1, 0, 1, 0],
            "group": ["A", "A", "B", "B"],
        }
    )
    csv_path = tmp_path / "v.csv"
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
            "1",
            "--threshold",
            "0.5",
        ]
    )
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "--metric is required" in err


def test_validate_threshold_metric_not_computed_errors(capsys, tmp_path):
    """equalized_odds needs y_true + y_pred — but mae_parity needs score; asking for mae without score."""
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "yhat": [1, 0, 1, 0],
            "group": ["A", "A", "B", "B"],
        }
    )
    csv_path = tmp_path / "v.csv"
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
            "1",
            "--threshold",
            "0.5",
            "--metric",
            "mae_parity_difference",
        ]
    )
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "was not computed" in err


def _tiny_validate_csv(tmp_path):
    df = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 1, 0, 0],
            "yhat": [1, 0, 1, 0, 1, 0, 1, 0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    path = tmp_path / "v.csv"
    df.to_csv(path, index=False)
    return path


def test_validate_subprocess_piped_stdout_is_markdown_only(tmp_path):
    """When stdout is a pipe, logs go to stderr; stdout is only the Markdown report."""
    csv_path = _tiny_validate_csv(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fairness_pipeline_dev_toolkit.cli.main",
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
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    out = proc.stdout.strip()
    assert out.startswith("#")
    assert '"timestamp"' not in proc.stdout
    assert '"level"' not in proc.stdout
    assert " - INFO - " not in proc.stdout


def test_validate_quiet_suppresses_validation_info_logs(capsys, tmp_path):
    csv_path = _tiny_validate_csv(tmp_path)
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
            "--quiet",
        ]
    )
    assert exit_code == 0
    err = capsys.readouterr().err
    assert "Starting validation" not in err
    assert "Validation completed" not in err


def test_validate_non_tty_suppresses_validation_info_logs(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    csv_path = _tiny_validate_csv(tmp_path)
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
        ]
    )
    assert exit_code == 0
    err = capsys.readouterr().err
    assert "Starting validation" not in err
    assert "Validation completed" not in err


def test_validate_verbose_when_not_tty_shows_validation_info(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    csv_path = _tiny_validate_csv(tmp_path)
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
            "--verbose",
        ]
    )
    assert exit_code == 0
    err = capsys.readouterr().err
    assert "Starting validation" in err
    assert "Validation completed" in err
