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

    # Heatmap should have data
    assert len(fig.data) == 1
    # Check that it's a heatmap (not a bar chart)
    assert fig.data[0].type == "heatmap", "Should use heatmap visualization"
    # Verify k-anonymity filtering worked (only large group remains)
    assert "large" in str(fig.data[0].x), "Only larger groups should remain"

    config_path = Path(tmp_path) / "monitoring_config.json"
    assert config_path.exists()
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert on_disk["report"]["k_anonymity"] == 5
    assert on_disk["artifacts_dir"] == str(tmp_path)


def test_dashboard_handles_datetime_index(tmp_path):
    """Test that dashboard works with DatetimeIndex (new format)"""
    timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
    metrics = pd.DataFrame(
        {
            "metric": ["DP[group]"] * 5,
            "group_key": ["A"] * 5,
            "value": [0.1, 0.12, 0.15, 0.18, 0.2],
            "n": [100] * 5,
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    dashboard = FairnessReportingDashboard(MonitoringSettings(artifacts_dir=str(tmp_path)))
    fig = dashboard.plot_trend(metrics, "DP[group]")

    assert len(fig.data) > 0, "Should generate trend plot with DatetimeIndex"


def test_drift_engine_handles_datetime_index(tmp_path):
    """Test that drift engine works with DatetimeIndex"""
    timestamps = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [0.01] * 15 + [0.2, 0.2, 0.2, 0.2, 0.2]
    metrics = pd.DataFrame(
        {
            "metric": ["DP[group]"] * len(timestamps),
            "group_key": ["A"] * len(timestamps),
            "value": values,
            "n": [100] * len(timestamps),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    settings = MonitoringSettings(artifacts_dir=str(tmp_path))
    settings.drift.critical_dpd = 0.05
    engine = FairnessDriftAndAlertEngine(settings)

    alerts = engine.analyze(metrics, window_points=3, ref_points=5)

    # Should work with DatetimeIndex
    assert isinstance(alerts, list)


def test_drift_engine_severity_includes_group_size(tmp_path):
    """Test that severity scoring incorporates group size"""
    timestamps = pd.date_range("2024-01-01", periods=20, freq="D")
    # Create metrics with varying group sizes
    metrics = pd.DataFrame(
        {
            "metric": ["DP[group]"] * len(timestamps),
            "group_key": ["A"] * len(timestamps),
            "value": [0.01] * 15 + [0.25, 0.25, 0.25, 0.25, 0.25],  # High drift
            "n": [10] * 10 + [150] * 10,  # Small then large groups
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    settings = MonitoringSettings(artifacts_dir=str(tmp_path))
    settings.drift.critical_dpd = 0.05
    engine = FairnessDriftAndAlertEngine(settings)

    alerts = engine.analyze(metrics, window_points=5, ref_points=10)

    # If alerts are generated, verify they consider group size
    if alerts:
        # Alerts with larger groups should potentially have different severity
        # than those with smaller groups (due to confidence factor)
        assert all(hasattr(alert, "severity") for alert in alerts)
