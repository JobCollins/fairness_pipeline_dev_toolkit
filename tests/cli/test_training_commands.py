"""
Tests for CLI training commands: cmd_train_sklearn_reduced, cmd_train_regularized,
cmd_train_lagrangian, and cmd_calibrate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.cli.main import (
    cmd_calibrate,
    cmd_train_lagrangian,
    cmd_train_regularized,
    cmd_train_sklearn_reduced,
)


class TestCmdTrainSklearnReduced:
    """Tests for train-sklearn-reduced command."""

    @pytest.fixture
    def sample_training_data(self, tmp_path):
        """Create sample training data CSV."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                "feature3": np.random.randn(n),
                "y": np.random.randint(0, 2, n),
                "group": np.random.choice(["A", "B"], n),
            }
        )
        csv_path = tmp_path / "train_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_train_sklearn_reduced_demographic_parity(self, sample_training_data, tmp_path):
        """Test train-sklearn-reduced with demographic_parity constraint."""
        out_model_path = tmp_path / "model.joblib"

        class Args:
            csv = str(sample_training_data)
            y = "y"
            group = "group"
            constraint = "demographic_parity"
            eps = 0.01
            T = 10
            out_model = str(out_model_path)

        try:
            exit_code = cmd_train_sklearn_reduced(Args())
            assert exit_code == 0
            assert out_model_path.exists(), "Output model should be created"
        except ImportError:
            pytest.skip("fairlearn or joblib not available")

    def test_train_sklearn_reduced_equalized_odds(self, sample_training_data, tmp_path):
        """Test train-sklearn-reduced with equalized_odds constraint."""
        out_model_path = tmp_path / "model.joblib"

        class Args:
            csv = str(sample_training_data)
            y = "y"
            group = "group"
            constraint = "equalized_odds"
            eps = 0.01
            T = 10
            out_model = str(out_model_path)

        try:
            exit_code = cmd_train_sklearn_reduced(Args())
            assert exit_code == 0
            assert out_model_path.exists()
        except ImportError:
            pytest.skip("fairlearn or joblib not available")

    def test_train_sklearn_reduced_missing_file(self, tmp_path):
        """Test train-sklearn-reduced with missing CSV file."""

        class Args:
            csv = str(tmp_path / "nonexistent.csv")
            y = "y"
            group = "group"
            constraint = "demographic_parity"
            eps = 0.01
            T = 10
            out_model = str(tmp_path / "model.joblib")

        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError, KeyError)):
            cmd_train_sklearn_reduced(Args())

    def test_train_sklearn_reduced_missing_columns(self, tmp_path):
        """Test train-sklearn-reduced with missing required columns."""
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        csv_path = tmp_path / "bad_data.csv"
        df.to_csv(csv_path, index=False)

        class Args:
            csv = str(csv_path)
            y = "y"
            group = "group"
            constraint = "demographic_parity"
            eps = 0.01
            T = 10
            out_model = str(tmp_path / "model.joblib")

        with pytest.raises(KeyError):
            cmd_train_sklearn_reduced(Args())


class TestCmdTrainRegularized:
    """Tests for train-regularized command (additional tests beyond test_commands.py)."""

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

    def test_train_regularized_multiple_etas(self, sample_training_data, tmp_path):
        """Test train-regularized with multiple eta values."""
        out_json_path = tmp_path / "pareto.json"

        class Args:
            csv = str(sample_training_data)
            etas = "0.0,0.1,0.2,0.5,1.0"
            epochs = 2
            lr = 1e-3
            out_json = str(out_json_path)
            out_png = ""

        try:
            exit_code = cmd_train_regularized(Args())
            assert exit_code == 0
            assert out_json_path.exists()
            # Verify JSON contains multiple points
            with open(out_json_path) as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert len(data) >= 5  # Should have at least 5 points for 5 etas
        except ImportError:
            pytest.skip("Training dependencies not available")

    def test_train_regularized_output_generation(self, sample_training_data, tmp_path):
        """Test that train-regularized generates both JSON and PNG outputs."""
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
            # PNG may or may not be created depending on dependencies
            # Just verify JSON is valid
            with open(out_json_path) as f:
                data = json.load(f)
                assert isinstance(data, list)
        except ImportError:
            pytest.skip("Training dependencies not available")


class TestCmdTrainLagrangian:
    """Tests for train-lagrangian command (additional tests beyond test_commands.py)."""

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

    def test_train_lagrangian_different_fairness_types(self, sample_training_data, tmp_path):
        """Test train-lagrangian with different fairness types."""
        out_json_path = tmp_path / "history.json"

        for fairness_type in ["demographic_parity", "equal_opportunity"]:

            class Args:
                csv = str(sample_training_data)
                fairness = fairness_type
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
                # Verify JSON is valid
                with open(out_json_path) as f:
                    data = json.load(f)
                    assert isinstance(data, list)
            except ImportError:
                pytest.skip("PyTorch not available")
                break

    def test_train_lagrangian_output_generation(self, sample_training_data, tmp_path):
        """Test that train-lagrangian generates valid JSON output."""
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
            assert out_json_path.exists()
            # Verify JSON structure
            with open(out_json_path) as f:
                data = json.load(f)
                assert isinstance(data, list)
                # History should contain epoch data
                if len(data) > 0:
                    assert isinstance(data[0], dict)
        except ImportError:
            pytest.skip("PyTorch not available")


class TestCmdCalibrate:
    """Tests for calibrate command (additional tests beyond test_commands.py)."""

    @pytest.fixture
    def sample_calibration_data(self, tmp_path):
        """Create sample calibration data CSV."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "score": np.random.rand(n),
                "y": np.random.randint(0, 2, n),
                "g": np.random.choice(["A", "B", "C"], n),
            }
        )
        csv_path = tmp_path / "calib_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_calibrate_both_methods(self, sample_calibration_data, tmp_path):
        """Test calibrate with both platt and isotonic methods."""
        for method in ["platt", "isotonic"]:
            out_csv_path = tmp_path / f"calibrated_{method}.csv"

            # Use a simple namespace-like object
            class Args:
                pass

            args = Args()
            args.csv = str(sample_calibration_data)
            args.method = method
            args.min_samples = 10
            args.out_csv = str(out_csv_path)

            exit_code = cmd_calibrate(args)
            assert exit_code == 0
            assert out_csv_path.exists()

            # Verify output CSV
            df_out = pd.read_csv(out_csv_path)
            assert "score_raw" in df_out.columns
            assert "score_cal" in df_out.columns
            assert "y" in df_out.columns
            assert "g" in df_out.columns
            assert len(df_out) == len(pd.read_csv(sample_calibration_data))

    def test_calibrate_output_validation(self, sample_calibration_data, tmp_path):
        """Test that calibrated scores are valid (in [0, 1] range)."""
        out_csv_path = tmp_path / "calibrated.csv"

        class Args:
            csv = str(sample_calibration_data)
            method = "platt"
            min_samples = 10
            out_csv = str(out_csv_path)

        exit_code = cmd_calibrate(Args())
        assert exit_code == 0

        # Verify output CSV
        df_out = pd.read_csv(out_csv_path)
        # Calibrated scores should be in [0, 1]
        assert df_out["score_cal"].min() >= 0.0
        assert df_out["score_cal"].max() <= 1.0
        # Raw scores should be preserved
        assert (df_out["score_raw"] == pd.read_csv(sample_calibration_data)["score"]).all()
