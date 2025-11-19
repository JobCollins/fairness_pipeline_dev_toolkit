"""
End-to-end test for the integrated run-pipeline CLI command.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sample_data_csv(tmp_path: Path):
    """Create sample CSV data for testing."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "f0,f1,f2,sensitive,y\n"
        "0.1,0.2,0.3,A,0\n"
        "0.2,0.3,0.4,A,1\n"
        "0.3,0.4,0.5,B,0\n"
        "0.4,0.5,0.6,B,1\n"
        "0.15,0.25,0.35,A,0\n"
        "0.25,0.35,0.45,A,1\n"
        "0.35,0.45,0.55,B,0\n"
        "0.45,0.55,0.65,B,1\n"
        "0.12,0.22,0.32,A,0\n"
        "0.22,0.32,0.42,A,1\n"
        "0.32,0.42,0.52,B,0\n"
        "0.42,0.52,0.62,B,1\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def integrated_config(tmp_path: Path):
    """Create integrated config with training section."""
    config_path = tmp_path / "config.yml"
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
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def test_cli_run_pipeline_e2e(tmp_path: Path, sample_data_csv, integrated_config):
    """End-to-end test of run-pipeline CLI command."""
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "fairness_pipeline_dev_toolkit.cli.main",
        "run-pipeline",
        "--config",
        str(integrated_config),
        "--csv",
        str(sample_data_csv),
        "--output-dir",
        str(output_dir),
        "--min-group-size",
        "2",
    ]

    # Set PYTHONPATH to include the project root
    import os

    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, text=True, capture_output=True, cwd=str(project_root), env=env)

    # CLI may fail if training dependencies are not available
    # Check if it's a dependency error or actual failure
    if result.returncode != 0:
        # If it's a dependency error, that's expected - skip the artifact checks
        if (
            "fairlearn" in result.stderr.lower()
            or "ImportError" in result.stderr
            or "ModuleNotFoundError" in result.stderr
        ):
            pytest.skip("Training dependencies not available")
        # Validation failure (exit code 1) is acceptable - workflow completed
        if result.returncode == 1 and "WORKFLOW RESULTS" in result.stdout:
            # Workflow completed but validation failed - this is acceptable
            pass
        else:
            # Otherwise, it's a real failure
            assert False, f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Check that artifacts were created (workflow should complete even if validation fails)
    workflow_json = output_dir / "workflow_results.json"
    assert (
        workflow_json.exists()
    ), f"Workflow results should be created. STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    payload = json.loads(workflow_json.read_text(encoding="utf-8"))
    assert "baseline_metrics" in payload
    assert "final_metrics" in payload
    assert "validation_result" in payload

    # Check transformed data exists
    transformed_csv = output_dir / "transformed_data.csv"
    assert transformed_csv.exists(), "Transformed data CSV should be created"


def test_cli_run_pipeline_requires_training_section(tmp_path: Path, sample_data_csv):
    """Test that run-pipeline requires training section in config."""
    # Config without training section
    config_path = tmp_path / "config_no_training.yml"
    config_path.write_text(
        """
sensitive: ["sensitive"]
pipeline: []
""",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "fairness_pipeline_dev_toolkit.cli.main",
        "run-pipeline",
        "--config",
        str(config_path),
        "--csv",
        str(sample_data_csv),
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    assert result.returncode == 1
    assert "training" in result.stdout.lower() or "training" in result.stderr.lower()


def test_cli_run_pipeline_prints_results(tmp_path: Path, sample_data_csv, integrated_config):
    """Test that run-pipeline prints workflow results."""
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "fairness_pipeline_dev_toolkit.cli.main",
        "run-pipeline",
        "--config",
        str(integrated_config),
        "--csv",
        str(sample_data_csv),
        "--output-dir",
        str(output_dir),
        "--min-group-size",
        "2",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    # Should print workflow results
    assert "WORKFLOW RESULTS" in result.stdout or "Validation" in result.stdout
