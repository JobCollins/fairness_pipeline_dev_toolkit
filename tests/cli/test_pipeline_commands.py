"""
Tests for CLI pipeline command: cmd_pipeline_run().
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.cli.main import cmd_pipeline_run


class TestCmdPipelineRun:
    """Tests for pipeline-run command."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample CSV data."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "sensitive": ["A", "A", "B", "B", "A"],
            }
        )
        csv_path = tmp_path / "input.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a sample pipeline config file."""
        config_content = """
sensitive: ["sensitive"]
alpha: 0.05
proxy_threshold: 0.30
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {}
"""
        config_path = tmp_path / "pipeline.config.yml"
        config_path.write_text(config_content)
        return config_path

    def test_pipeline_run_with_detectors_enabled(
        self, sample_data, sample_config, tmp_path, capsys
    ):
        """Test pipeline run with detectors enabled."""
        out_csv_path = tmp_path / "output.csv"
        detector_json_path = tmp_path / "detectors.json"
        report_md_path = tmp_path / "report.md"

        class Args:
            config = str(sample_config)
            csv = str(sample_data)
            out_csv = str(out_csv_path)
            detector_json = str(detector_json_path)
            report_md = str(report_md_path)
            no_detectors = False
            profile = None

        exit_code = cmd_pipeline_run(Args())

        assert exit_code == 0
        assert out_csv_path.exists(), "Output CSV should be created"
        assert detector_json_path.exists(), "Detector JSON should be created"
        assert report_md_path.exists(), "Report markdown should be created"

        # Verify detector JSON is valid
        import json

        with open(detector_json_path) as f:
            detector_data = json.load(f)
            assert "meta" in detector_data or "body" in detector_data

        # Verify output CSV
        output_df = pd.read_csv(out_csv_path)
        assert len(output_df) > 0

        # Check that detector summary was printed
        captured = capsys.readouterr()
        assert "Detector Summary" in captured.out or "Pipeline completed" in captured.out

    def test_pipeline_run_with_detectors_disabled(self, sample_data, sample_config, tmp_path):
        """Test pipeline run with detectors disabled."""
        out_csv_path = tmp_path / "output.csv"

        class Args:
            config = str(sample_config)
            csv = str(sample_data)
            out_csv = str(out_csv_path)
            detector_json = None
            report_md = None
            no_detectors = True
            profile = None

        exit_code = cmd_pipeline_run(Args())

        assert exit_code == 0
        assert out_csv_path.exists(), "Output CSV should be created"

    def test_pipeline_run_output_generation(self, sample_data, sample_config, tmp_path):
        """Test that all output files are generated correctly."""
        out_csv_path = tmp_path / "output.csv"
        detector_json_path = tmp_path / "detectors.json"
        report_md_path = tmp_path / "report.md"

        class Args:
            config = str(sample_config)
            csv = str(sample_data)
            out_csv = str(out_csv_path)
            detector_json = str(detector_json_path)
            report_md = str(report_md_path)
            no_detectors = False
            profile = None

        exit_code = cmd_pipeline_run(Args())

        assert exit_code == 0

        # Verify CSV output
        output_df = pd.read_csv(out_csv_path)
        assert len(output_df) > 0

        # Verify detector JSON
        import json

        with open(detector_json_path) as f:
            detector_data = json.load(f)
            assert isinstance(detector_data, dict)

        # Verify markdown report
        report_content = report_md_path.read_text()
        assert "Pipeline Run Report" in report_content
        assert "Config" in report_content

    def test_pipeline_run_missing_config(self, sample_data, tmp_path, capsys):
        """Test error handling when config is missing."""

        class Args:
            config = None
            csv = str(sample_data)
            out_csv = str(tmp_path / "output.csv")
            detector_json = None
            report_md = None
            no_detectors = False
            profile = None

        # Unset environment variable if it exists
        old_env = os.environ.pop("FAIRPIPE_CONFIG_PATH", None)
        try:
            exit_code = cmd_pipeline_run(Args())
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "Error" in captured.out or "config" in captured.out.lower()
        finally:
            if old_env:
                os.environ["FAIRPIPE_CONFIG_PATH"] = old_env

    def test_pipeline_run_invalid_csv(self, sample_config, tmp_path, capsys):
        """Test error handling with invalid CSV file."""
        invalid_csv = tmp_path / "nonexistent.csv"

        class Args:
            config = str(sample_config)
            csv = str(invalid_csv)
            out_csv = str(tmp_path / "output.csv")
            detector_json = None
            report_md = None
            no_detectors = False
            profile = None

        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            cmd_pipeline_run(Args())

    def test_pipeline_run_with_profile(self, sample_data, tmp_path):
        """Test pipeline run with profile selection."""
        # Create config with profiles
        config_content = """
sensitive: ["sensitive"]
alpha: 0.05
proxy_threshold: 0.30
profiles:
  pipeline:
    pipeline:
      - name: reweigh
        transformer: "InstanceReweighting"
        params: {}
"""
        config_path = tmp_path / "pipeline.config.yml"
        config_path.write_text(config_content)

        out_csv_path = tmp_path / "output.csv"

        class Args:
            config = str(config_path)
            csv = str(sample_data)
            out_csv = str(out_csv_path)
            detector_json = None
            report_md = None
            no_detectors = True
            profile = "pipeline"

        exit_code = cmd_pipeline_run(Args())
        assert exit_code == 0
        assert out_csv_path.exists()
