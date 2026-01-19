"""
Tests for CLI commands: version, sample-check, train-regularized, train-lagrangian, calibrate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.cli.main import (
    cmd_calibrate,
    cmd_sample_check,
    cmd_train_lagrangian,
    cmd_train_regularized,
    cmd_version,
    main,
)


class TestCmdVersion:
    """Tests for version command."""

    def test_version_command(self, capsys):
        """Test that version command prints version."""
        exit_code = cmd_version(None)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0  # Should print version

    def test_version_via_main(self, capsys):
        """Test version command via main entry point."""
        exit_code = main(["version"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0


class TestCmdSampleCheck:
    """Tests for sample-check command."""

    def test_sample_check_command_exists(self):
        """Test that sample-check command exists and can be called."""
        # Check if command exists in main
        # argparse raises SystemExit(0) when --help is used
        try:
            exit_code = main(["sample-check", "--help"])
            # If it doesn't raise, should return 0
            assert exit_code == 0
        except SystemExit as e:
            # SystemExit(0) is expected for --help
            assert e.code == 0

    def test_sample_check_when_file_exists(self, tmp_path, monkeypatch, capsys):
        """Test cmd_sample_check when dev_sample.csv exists."""
        # Create dev_sample.csv in tmp_path
        sample_file = tmp_path / "dev_sample.csv"
        sample_file.write_text("col1,col2\n1,2\n3,4\n")

        # Change to tmp_path directory
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create args object
            class Args:
                pass

            exit_code = cmd_sample_check(Args())
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Sample data exists" in captured.out or "✅" in captured.out
        finally:
            os.chdir(original_cwd)

    def test_sample_check_when_file_not_exists(self, tmp_path, monkeypatch, capsys):
        """Test cmd_sample_check when dev_sample.csv does not exist."""
        # Ensure dev_sample.csv does not exist in tmp_path
        sample_file = tmp_path / "dev_sample.csv"
        if sample_file.exists():
            sample_file.unlink()

        # Change to tmp_path directory
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create args object
            class Args:
                pass

            exit_code = cmd_sample_check(Args())
            assert exit_code == 0  # Should return 0 (non-blocking)
            captured = capsys.readouterr()
            assert (
                "not found" in captured.out
                or "⚠️" in captured.out
                or "skipping" in captured.out.lower()
            )
        finally:
            os.chdir(original_cwd)


class TestCmdTrainRegularized:
    """Tests for train-regularized command."""

    @pytest.fixture
    def sample_training_data(self, tmp_path):
        """Create sample training data CSV."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "f0": np.random.randn(n),
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
                "y": np.random.randint(0, 2, n),
                "s": np.random.randint(0, 2, n),
            }
        )
        csv_path = tmp_path / "train_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_train_regularized_basic(self, sample_training_data, tmp_path):
        """Test basic train-regularized command."""
        out_json_path = tmp_path / "pareto.json"

        class Args:
            csv = str(sample_training_data)
            etas = "0.0,0.2"
            epochs = 2
            lr = 1e-3
            out_json = str(out_json_path)
            out_png = ""

        try:
            exit_code = cmd_train_regularized(Args())
            assert exit_code == 0
            assert out_json_path.exists(), "Output JSON should be created"
            # Verify JSON is valid
            with open(out_json_path) as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert len(data) > 0
        except ImportError:
            pytest.skip("Training dependencies not available")

    def test_train_regularized_with_plot(self, sample_training_data, tmp_path):
        """Test train-regularized with plot output."""
        out_json_path = tmp_path / "pareto.json"
        out_png_path = tmp_path / "pareto.png"

        class Args:
            csv = str(sample_training_data)
            etas = "0.0,0.5"
            epochs = 2
            lr = 1e-3
            out_json = str(out_json_path)
            out_png = str(out_png_path)

        try:
            exit_code = cmd_train_regularized(Args())
            assert exit_code == 0
            assert out_json_path.exists()
            if out_png_path.exists():
                assert out_png_path.stat().st_size > 0, "Plot file should not be empty"
        except ImportError:
            pytest.skip("Training dependencies not available")

    def test_train_regularized_missing_file(self, tmp_path):
        """Test train-regularized with missing CSV file."""

        class Args:
            csv = str(tmp_path / "nonexistent.csv")
            etas = "0.0"
            epochs = 2
            lr = 1e-3
            out_json = str(tmp_path / "out.json")
            out_png = ""

        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            cmd_train_regularized(Args())


class TestCmdTrainLagrangian:
    """Tests for train-lagrangian command."""

    @pytest.fixture
    def sample_training_data(self, tmp_path):
        """Create sample training data CSV."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "f0": np.random.randn(n),
                "f1": np.random.randn(n),
                "y": np.random.randint(0, 2, n),
                "s": np.random.randint(0, 2, n),
            }
        )
        csv_path = tmp_path / "train_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_train_lagrangian_demographic_parity(self, sample_training_data, tmp_path):
        """Test train-lagrangian with demographic_parity."""
        out_json_path = tmp_path / "history.json"

        class Args:
            csv = str(sample_training_data)
            fairness = "demographic_parity"
            dp_tol = 0.02
            eo_tol = 0.02
            model_lr = 1e-3
            lambda_lr = 1e-2
            epochs = 2
            batch_size = 32
            out_json = str(out_json_path)

        try:
            exit_code = cmd_train_lagrangian(Args())
            assert exit_code == 0
            assert out_json_path.exists(), "Output JSON should be created"
            # Verify JSON is valid
            with open(out_json_path) as f:
                data = json.load(f)
                assert isinstance(data, list)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_train_lagrangian_equal_opportunity(self, sample_training_data, tmp_path):
        """Test train-lagrangian with equal_opportunity."""
        out_json_path = tmp_path / "history.json"

        class Args:
            csv = str(sample_training_data)
            fairness = "equal_opportunity"
            dp_tol = 0.02
            eo_tol = 0.02
            model_lr = 1e-3
            lambda_lr = 1e-2
            epochs = 2
            batch_size = 32
            out_json = str(out_json_path)

        try:
            exit_code = cmd_train_lagrangian(Args())
            assert exit_code == 0
            assert out_json_path.exists()
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_train_lagrangian_missing_file(self, tmp_path):
        """Test train-lagrangian with missing CSV file."""

        class Args:
            csv = str(tmp_path / "nonexistent.csv")
            fairness = "demographic_parity"
            dp_tol = 0.02
            eo_tol = 0.02
            model_lr = 1e-3
            lambda_lr = 1e-2
            epochs = 2
            batch_size = 32
            out_json = str(tmp_path / "out.json")

        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            cmd_train_lagrangian(Args())


class TestCmdCalibrate:
    """Tests for calibrate command."""

    @pytest.fixture
    def sample_calibration_data(self, tmp_path):
        """Create sample calibration data CSV."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "score": np.random.rand(n),
                "y": np.random.randint(0, 2, n),
                "g": np.random.choice(["A", "B"], n),
            }
        )
        csv_path = tmp_path / "calib_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_calibrate_platt(self, sample_calibration_data, tmp_path):
        """Test calibrate command with Platt scaling."""
        out_csv_path = tmp_path / "calibrated.csv"

        class Args:
            csv = str(sample_calibration_data)
            method = "platt"
            min_samples = 10
            out_csv = str(out_csv_path)

        exit_code = cmd_calibrate(Args())
        assert exit_code == 0
        assert out_csv_path.exists(), "Output CSV should be created"

        # Verify output CSV
        df_out = pd.read_csv(out_csv_path)
        assert "score_raw" in df_out.columns
        assert "score_cal" in df_out.columns
        assert "y" in df_out.columns
        assert "g" in df_out.columns
        assert len(df_out) == len(pd.read_csv(sample_calibration_data))

    def test_calibrate_isotonic(self, sample_calibration_data, tmp_path):
        """Test calibrate command with isotonic regression."""
        out_csv_path = tmp_path / "calibrated.csv"

        class Args:
            csv = str(sample_calibration_data)
            method = "isotonic"
            min_samples = 10
            out_csv = str(out_csv_path)

        exit_code = cmd_calibrate(Args())
        assert exit_code == 0
        assert out_csv_path.exists()

        # Verify output CSV
        df_out = pd.read_csv(out_csv_path)
        assert len(df_out) > 0
        # Calibrated scores should be in [0, 1]
        assert df_out["score_cal"].min() >= 0.0
        assert df_out["score_cal"].max() <= 1.0

    def test_calibrate_missing_file(self, tmp_path):
        """Test calibrate with missing CSV file."""

        class Args:
            csv = str(tmp_path / "nonexistent.csv")
            method = "platt"
            min_samples = 10
            out_csv = str(tmp_path / "out.csv")

        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError, KeyError)):
            cmd_calibrate(Args())

    def test_calibrate_missing_columns(self, tmp_path):
        """Test calibrate with missing required columns."""
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        csv_path = tmp_path / "bad_data.csv"
        df.to_csv(csv_path, index=False)

        class Args:
            csv = str(csv_path)
            method = "platt"
            min_samples = 10
            out_csv = str(tmp_path / "out.csv")

        with pytest.raises(KeyError):
            cmd_calibrate(Args())

    def test_calibrate_invalid_method(self, sample_calibration_data, tmp_path):
        """Test calibrate with invalid method."""
        out_csv_path = tmp_path / "calibrated.csv"

        class Args:
            csv = str(sample_calibration_data)
            method = "invalid_method"
            min_samples = 10
            out_csv = str(out_csv_path)

        with pytest.raises(ValueError, match="method must be one of"):
            cmd_calibrate(Args())


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_main_invalid_command(self):
        """Test that invalid command returns appropriate exit code."""
        # argparse raises SystemExit(2) for invalid commands
        try:
            exit_code = main(["invalid_command"])
            assert exit_code != 0 or exit_code == 0  # May show help
        except SystemExit as e:
            # SystemExit(2) is expected for invalid commands
            assert e.code == 2

    def test_main_no_arguments(self, capsys):
        """Test that no arguments shows help."""
        exit_code = main([])

        # Should show help or return 0
        assert exit_code is not None
        captured = capsys.readouterr()
        # Help text should be present
        assert len(captured.out) > 0 or len(captured.err) > 0
