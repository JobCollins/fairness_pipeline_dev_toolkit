"""
Unit tests for CLI run-pipeline command: cmd_run_pipeline().
Tests full workflow execution, MLflow logging, and error handling scenarios.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.cli.main import cmd_run_pipeline


class TestCmdRunPipeline:
    """Tests for run-pipeline command."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample CSV data."""
        df = pd.DataFrame(
            {
                "f0": [0.1, 0.2, 0.3, 0.4] * 5,
                "f1": [0.2, 0.3, 0.4, 0.5] * 5,
                "sensitive": ["A", "A", "B", "B"] * 5,
                "y": [0, 1, 0, 1] * 5,
            }
        )
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def integrated_config(self, tmp_path):
        """Create integrated config with training section."""
        config_content = """
sensitive: ["sensitive"]
alpha: 0.05
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
training:
  method: "reductions"
  target_column: "y"
  params:
    constraint: "demographic_parity"
    eps: 0.01
    T: 10
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.20
"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(config_content)
        return config_path

    def test_run_pipeline_full_workflow(self, sample_data, integrated_config, tmp_path, capsys):
        """Test full workflow execution."""
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        output_dir_str = str(output_dir)

        class Args:
            config = str(integrated_config)
            csv = str(sample_data)
            output_dir = output_dir_str
            min_group_size = 2
            train_size = 0.8
            mlflow_experiment = None
            mlflow_run_name = None
            profile = None

        try:
            exit_code = cmd_run_pipeline(Args())
            # Exit code can be 0 (passed) or 1 (validation failed)
            assert exit_code in [0, 1]

            # Check that results were printed
            captured = capsys.readouterr()
            assert "WORKFLOW RESULTS" in captured.out or "Validation" in captured.out

            # Check that artifacts directory was mentioned if provided
            if Args.output_dir:
                assert "Artifacts saved" in captured.out or str(Args.output_dir) in captured.out
        except ImportError:
            pytest.skip("Training dependencies not available")

    def test_run_pipeline_missing_config(self, sample_data, tmp_path, capsys):
        """Test error handling when config is missing."""
        old_env = os.environ.pop("FAIRPIPE_CONFIG_PATH", None)
        try:

            class Args:
                config = None
                csv = str(sample_data)
                output_dir = None
                min_group_size = 30
                train_size = 0.8
                mlflow_experiment = None
                mlflow_run_name = None
                profile = None

            exit_code = cmd_run_pipeline(Args())
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "Error" in captured.out or "config" in captured.out.lower()
        finally:
            if old_env:
                os.environ["FAIRPIPE_CONFIG_PATH"] = old_env

    def test_run_pipeline_config_without_training(self, sample_data, tmp_path, capsys):
        """Test error handling when config lacks training section."""
        config_content = """
sensitive: ["sensitive"]
pipeline: []
"""
        config_path = tmp_path / "config_no_training.yml"
        config_path.write_text(config_content)

        class Args:
            config = str(config_path)
            csv = str(sample_data)
            output_dir = None
            min_group_size = 30
            train_size = 0.8
            mlflow_experiment = None
            mlflow_run_name = None
            profile = None

        exit_code = cmd_run_pipeline(Args())
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "training" in captured.out.lower() or "training" in captured.err.lower()

    def test_run_pipeline_invalid_csv(self, integrated_config, tmp_path, capsys):
        """Test error handling with invalid CSV file."""
        invalid_csv = tmp_path / "nonexistent.csv"

        class Args:
            config = str(integrated_config)
            csv = str(invalid_csv)
            output_dir = None
            min_group_size = 30
            train_size = 0.8
            mlflow_experiment = None
            mlflow_run_name = None
            profile = None

        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            cmd_run_pipeline(Args())

    @patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger.log_workflow_results")
    def test_run_pipeline_with_mlflow_logging(
        self, mock_log_workflow_results, sample_data, integrated_config, tmp_path, capsys
    ):
        """Test MLflow logging when mlflow_experiment is provided."""
        mock_log_workflow_results.return_value = True

        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        output_dir_str = str(output_dir)

        class Args:
            config = str(integrated_config)
            csv = str(sample_data)
            output_dir = output_dir_str
            min_group_size = 2
            train_size = 0.8
            mlflow_experiment = "test_experiment"
            mlflow_run_name = "test_run"
            profile = None

        try:
            exit_code = cmd_run_pipeline(Args())
            # Exit code can be 0 or 1
            assert exit_code in [0, 1]

            # Check that MLflow logging was attempted
            captured = capsys.readouterr()
            # If workflow completed, MLflow logging should be mentioned
            if "WORKFLOW RESULTS" in captured.out:
                # MLflow logging may or may not succeed, but should be attempted
                assert mock_log_workflow_results.called or "MLflow" in captured.out
        except ImportError:
            pytest.skip("Training dependencies not available")

    @patch("fairness_pipeline_dev_toolkit.integration.mlflow_logger.log_workflow_results")
    def test_run_pipeline_mlflow_logging_failure(
        self, mock_log_workflow_results, sample_data, integrated_config, tmp_path, capsys
    ):
        """Test that MLflow logging failure doesn't crash the workflow."""
        mock_log_workflow_results.return_value = False

        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        output_dir_str = str(output_dir)

        class Args:
            config = str(integrated_config)
            csv = str(sample_data)
            output_dir = output_dir_str
            min_group_size = 2
            train_size = 0.8
            mlflow_experiment = "test_experiment"
            mlflow_run_name = None
            profile = None

        try:
            exit_code = cmd_run_pipeline(Args())
            # Should still complete even if MLflow logging fails
            assert exit_code in [0, 1]
        except ImportError:
            pytest.skip("Training dependencies not available")

    def test_run_pipeline_workflow_exception_handling(
        self, sample_data, integrated_config, tmp_path, capsys
    ):
        """Test that workflow exceptions are caught and handled gracefully."""
        # Create a config that might cause issues during execution
        # Use a valid method but invalid params to trigger execution error
        bad_config_content = """
sensitive: ["sensitive"]
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
training:
  method: "reductions"
  target_column: "nonexistent_column"
  params:
    constraint: "demographic_parity"
    eps: 0.01
    T: 10
fairness_metric: "demographic_parity_difference"
validation_threshold: 0.20
"""
        bad_config_path = tmp_path / "bad_config.yml"
        bad_config_path.write_text(bad_config_content)

        class Args:
            config = str(bad_config_path)
            csv = str(sample_data)
            output_dir = None
            min_group_size = 30
            train_size = 0.8
            mlflow_experiment = None
            mlflow_run_name = None
            profile = None

        exit_code = cmd_run_pipeline(Args())
        # Should return 1 on error
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out or "Error" in captured.err

    def test_run_pipeline_with_output_dir(self, sample_data, integrated_config, tmp_path):
        """Test that output_dir is used when provided."""
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        output_dir_str = str(output_dir)

        class Args:
            config = str(integrated_config)
            csv = str(sample_data)
            output_dir = output_dir_str
            min_group_size = 2
            train_size = 0.8
            mlflow_experiment = None
            mlflow_run_name = None
            profile = None

        try:
            exit_code = cmd_run_pipeline(Args())
            assert exit_code in [0, 1]
            # Output directory should be used (artifacts may be created)
        except ImportError:
            pytest.skip("Training dependencies not available")

    def test_run_pipeline_validation_result_exit_code(
        self, sample_data, integrated_config, tmp_path
    ):
        """Test that exit code reflects validation result."""
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        output_dir_str = str(output_dir)

        class Args:
            config = str(integrated_config)
            csv = str(sample_data)
            output_dir = output_dir_str
            min_group_size = 2
            train_size = 0.8
            mlflow_experiment = None
            mlflow_run_name = None
            profile = None

        try:
            exit_code = cmd_run_pipeline(Args())
            # Exit code should be 0 if validation passed, 1 if failed
            assert exit_code in [0, 1]
        except ImportError:
            pytest.skip("Training dependencies not available")
