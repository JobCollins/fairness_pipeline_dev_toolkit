# tests/system/test_cli_e2e_pipeline.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "backend", ["native"]
)  # keep simple; measurement backends not required here
def test_cli_pipeline_e2e(tmp_path: Path, backend):
    """
    End-to-end smoke of the Pipeline CLI:
      - writes a small CSV and YAML config
      - runs: python -m fairness_pipeline_dev_toolkit.cli.main pipeline ...
      - asserts artifacts exist and detector JSON has expected keys
    """
    # --- prepare tiny CSV ---
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "group,x1,x2,y\n"
        "A,0.10,1,0\n"
        "A,0.20,2,1\n"
        "B,0.30,3,0\n"
        "B,0.40,3,1\n"
        "A,0.15,2,0\n",
        encoding="utf-8",
    )

    # --- prepare YAML config (compatible with PipelineConfig & engine._make_step) ---
    # NOTE: 'pipeline' uses your transformer names registered in _TRANSFORMER_REGISTRY
    # InstanceReweighting pulls defaults from cfg.sensitive/benchmarks if params omitted.
    # DisparateImpactRemover requires features list and a single 'sensitive'.
    yml = f"""
sensitive: ["group"]
alpha: 0.05
proxy_threshold: 0.30
report_out: "{(tmp_path / 'detectors.json').as_posix()}"
benchmarks:
  group: {{A: 0.5, B: 0.5}}
pipeline:
  - name: reweigh
    transformer: "InstanceReweighting"
    params: {{}}
  - name: di
    transformer: "DisparateImpactRemover"
    params:
      features: ["x1"]
      sensitive: "group"
      repair_level: 0.8
"""
    cfg_path = tmp_path / "pipeline.yml"
    cfg_path.write_text(yml, encoding="utf-8")

    out_csv = tmp_path / "out.csv"
    md_report = tmp_path / "run.md"

    # --- run CLI ---
    cmd = [
        sys.executable,
        "-m",
        "fairness_pipeline_dev_toolkit.cli.main",
        "pipeline",
        "--config",
        str(cfg_path),
        "--csv",
        str(csv_path),
        "--out-csv",
        str(out_csv),
        "--detector-json",
        str(tmp_path / "detectors.json"),
        "--report-md",
        str(md_report),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert (
        result.returncode == 0
    ), f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # --- artifacts exist ---
    assert out_csv.exists() and out_csv.stat().st_size > 0
    det_json = tmp_path / "detectors.json"
    assert det_json.exists() and det_json.stat().st_size > 0
    assert md_report.exists() and md_report.stat().st_size > 0

    # --- detector JSON structure check ---
    payload = json.loads(det_json.read_text(encoding="utf-8"))
    assert "meta" in payload and "body" in payload
    body = payload["body"]
    for key in ("summary", "representation", "disparities", "proxies"):
        assert key in body, f"missing '{key}' in detector report body"
    assert isinstance(body["representation"], list)
    assert isinstance(body["disparities"], list)
    assert isinstance(body["proxies"], list)
