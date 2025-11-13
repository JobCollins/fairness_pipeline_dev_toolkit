from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.monitoring import (
    FairnessDriftAndAlertEngine,
    FairnessReportingDashboard,
    MonitoringSettings,
    ReportConfig,
)

pytest.importorskip("plotly")


def _make_metrics_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [0.01] * 7 + [0.2, 0.2, 0.2]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "metric": ["DP[group]"] * len(timestamps),
            "group_key": ["A"] * len(timestamps),
            "value": values,
            "n": [100] * len(timestamps),
        }
    )


def test_drift_engine_uses_settings_threshold(tmp_path):
    metrics = _make_metrics_frame()
    settings = MonitoringSettings(artifacts_dir=str(tmp_path))
    settings.drift.critical_dpd = 0.05
    engine = FairnessDriftAndAlertEngine(settings)

    alerts = engine.analyze(metrics, window_points=3, ref_points=5)

    assert alerts, "Expected drift alert when critical_dpd is low"
    assert alerts[0].metric == "DP[group]"


def test_dashboard_applies_k_anonymity_and_persists_config(tmp_path):
    metrics = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "metric": ["DP[group]", "DP[group]"],
            "group_key": ["small", "large"],
            "value": [0.12, 0.18],
            "n": [3, 7],
        }
    )

    settings = MonitoringSettings(
        artifacts_dir=str(tmp_path),
        report=ReportConfig(k_anonymity=5),
    )
    dashboard = FairnessReportingDashboard(settings)
    fig = dashboard.plot_intersectional(metrics, "DP[")

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["large"], "Only larger groups should remain"

    config_path = Path(tmp_path) / "monitoring_config.json"
    assert config_path.exists()
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert on_disk["report"]["k_anonymity"] == 5
    assert on_disk["artifacts_dir"] == str(tmp_path)
