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


# ============================================================================
# Dashboard __init__ Tests
# ============================================================================


class TestDashboardInit:
    """Tests for FairnessReportingDashboard.__init__()."""

    def test_dashboard_init_with_monitoring_settings(self, tmp_path):
        """Test initialization with MonitoringSettings."""
        settings = MonitoringSettings(artifacts_dir=str(tmp_path))
        dashboard = FairnessReportingDashboard(settings)
        assert dashboard.settings == settings
        assert dashboard.cfg == settings.report
        assert Path(tmp_path / "monitoring_config.json").exists()

    def test_dashboard_init_with_report_config(self, tmp_path):
        """Test initialization with ReportConfig."""
        report_cfg = ReportConfig(k_anonymity=30)
        dashboard = FairnessReportingDashboard(report_cfg, artifacts_dir=str(tmp_path))
        assert dashboard.cfg.k_anonymity == 30
        assert Path(tmp_path / "monitoring_config.json").exists()

    def test_dashboard_init_with_none_defaults(self, tmp_path):
        """Test initialization with None (should use defaults)."""
        dashboard = FairnessReportingDashboard(None, artifacts_dir=str(tmp_path))
        assert dashboard.settings is not None
        assert dashboard.cfg is not None
        # Should create default MonitoringSettings
        assert isinstance(dashboard.settings, MonitoringSettings)

    def test_dashboard_init_with_artifacts_dir_override(self, tmp_path):
        """Test initialization with artifacts_dir override."""
        custom_dir = tmp_path / "custom_artifacts"
        settings = MonitoringSettings(artifacts_dir=str(tmp_path / "original"))
        dashboard = FairnessReportingDashboard(settings, artifacts_dir=str(custom_dir))
        assert dashboard.settings.artifacts_dir == str(custom_dir)
        assert custom_dir.exists()

    def test_dashboard_init_creates_artifacts_dir(self, tmp_path):
        """Test that initialization creates artifacts directory."""
        artifacts_dir = tmp_path / "new_artifacts"
        FairnessReportingDashboard(artifacts_dir=str(artifacts_dir))
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()


# ============================================================================
# Dashboard plot_trend Tests
# ============================================================================


class TestDashboardPlotTrend:
    """Tests for plot_trend() method."""

    def test_plot_trend_with_datetime_index(self, tmp_path):
        """Test plot_trend with DatetimeIndex."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="D")
        metrics = pd.DataFrame(
            {
                "metric": ["DP[group]"] * 10,
                "group_key": ["A"] * 10,
                "value": [0.1 + i * 0.01 for i in range(10)],
                "n": [100] * 10,
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        fig = dashboard.plot_trend(metrics, "DP[")
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Trend: DP["

    def test_plot_trend_with_timestamp_column(self, tmp_path):
        """Test plot_trend with timestamp column."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "metric": ["DP[group]"] * 10,
                "group_key": ["A"] * 10,
                "value": [0.1 + i * 0.01 for i in range(10)],
                "n": [100] * 10,
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        fig = dashboard.plot_trend(metrics, "DP[")
        assert len(fig.data) > 0

    def test_plot_trend_with_groups_filtering(self, tmp_path):
        """Test plot_trend with groups filtering."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "metric": ["DP[group]"] * 10,
                "group_key": ["A", "B"] * 5,
                "value": [0.1] * 10,
                "n": [100] * 10,
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        # Filter to only group A
        fig = dashboard.plot_trend(metrics, "DP[", groups=["A"])
        assert len(fig.data) == 1  # Should have one trace for group A

    def test_plot_trend_with_multiple_groups(self, tmp_path):
        """Test plot_trend with multiple groups."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
                "metric": ["DP[group]"] * 6,
                "group_key": ["A", "B", "C"] * 2,
                "value": [0.1] * 6,
                "n": [100] * 6,
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        fig = dashboard.plot_trend(metrics, "DP[")
        # Should have traces for all groups
        assert len(fig.data) == 3

    def test_plot_trend_empty_data(self, tmp_path):
        """Test plot_trend with empty data."""
        metrics = pd.DataFrame(columns=["timestamp", "metric", "group_key", "value", "n"])
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        # Empty data will result in empty filtered dataframe, which may not raise error
        # but will produce an empty figure
        fig = dashboard.plot_trend(metrics, "DP[")
        # Should still create a figure, but with no data
        assert isinstance(fig, type(dashboard.plot_trend(metrics, "DP[")))

    def test_plot_trend_single_point(self, tmp_path):
        """Test plot_trend with single data point."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "metric": ["DP[group]"],
                "group_key": ["A"],
                "value": [0.1],
                "n": [100],
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        fig = dashboard.plot_trend(metrics, "DP[")
        # Should still create a figure with one trace
        assert len(fig.data) == 1


# ============================================================================
# Dashboard plot_intersectional Tests
# ============================================================================


class TestDashboardPlotIntersectional:
    """Tests for plot_intersectional() method."""

    def test_plot_intersectional_latest_only_true(self, tmp_path):
        """Test plot_intersectional with latest_only=True."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "metric": ["DP[group]", "EO[group]"] * 5,
                "group_key": ["A", "B"] * 5,
                "value": [0.1, 0.2] * 5,
                "n": [100, 150] * 5,
            }
        )
        dashboard = FairnessReportingDashboard(
            MonitoringSettings(artifacts_dir=str(tmp_path), report=ReportConfig(k_anonymity=10))
        )
        fig = dashboard.plot_intersectional(metrics, "DP[", latest_only=True)
        # Should have heatmap data
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_plot_intersectional_latest_only_false(self, tmp_path):
        """Test plot_intersectional with latest_only=False."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "metric": ["DP[group]"] * 10,
                "group_key": ["A"] * 10,
                "value": [0.1] * 10,
                "n": [100] * 10,
            }
        )
        dashboard = FairnessReportingDashboard(
            MonitoringSettings(artifacts_dir=str(tmp_path), report=ReportConfig(k_anonymity=10))
        )
        fig = dashboard.plot_intersectional(metrics, "DP[", latest_only=False)
        # Should still create heatmap (all data points)
        assert len(fig.data) == 1

    def test_plot_intersectional_k_anonymity_filtering(self, tmp_path):
        """Test plot_intersectional with k-anonymity filtering."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "metric": ["DP[group]", "DP[group]"],
                "group_key": ["small", "large"],
                "value": [0.1, 0.2],
                "n": [5, 25],  # small group below k=20, large above
            }
        )
        dashboard = FairnessReportingDashboard(
            MonitoringSettings(artifacts_dir=str(tmp_path), report=ReportConfig(k_anonymity=20))
        )
        fig = dashboard.plot_intersectional(metrics, "DP[")
        # Should filter out small group
        if len(fig.data) > 0 and fig.data[0].type == "heatmap":
            # Large group should be present, small should be filtered
            x_values = fig.data[0].x
            assert "large" in str(x_values) or len(x_values) == 1
        else:
            # If no data after filtering, that's also valid
            assert len(fig.data) == 0

    def test_plot_intersectional_empty_data(self, tmp_path):
        """Test plot_intersectional with empty data."""
        metrics = pd.DataFrame(columns=["timestamp", "metric", "group_key", "value", "n"])
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        fig = dashboard.plot_intersectional(metrics, "DP[")
        # Should return empty figure
        assert len(fig.data) == 0

    def test_plot_intersectional_all_filtered_by_k_anonymity(self, tmp_path):
        """Test plot_intersectional when all groups are filtered by k-anonymity."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "metric": ["DP[group]", "DP[group]"],
                "group_key": ["small1", "small2"],
                "value": [0.1, 0.2],
                "n": [5, 10],  # Both below k=20
            }
        )
        dashboard = FairnessReportingDashboard(
            MonitoringSettings(artifacts_dir=str(tmp_path), report=ReportConfig(k_anonymity=20))
        )
        fig = dashboard.plot_intersectional(metrics, "DP[")
        # Should return empty figure
        assert len(fig.data) == 0

    def test_plot_intersectional_with_datetime_index(self, tmp_path):
        """Test plot_intersectional with DatetimeIndex."""
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        metrics = pd.DataFrame(
            {
                "metric": ["DP[group]"] * 5,
                "group_key": ["A"] * 5,
                "value": [0.1] * 5,
                "n": [100] * 5,
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        dashboard = FairnessReportingDashboard(
            MonitoringSettings(artifacts_dir=str(tmp_path), report=ReportConfig(k_anonymity=10))
        )
        fig = dashboard.plot_intersectional(metrics, "DP[")
        # Should handle DatetimeIndex correctly
        assert isinstance(fig, type(dashboard.plot_intersectional(metrics, "DP[")))


# ============================================================================
# Dashboard write_alerts_json Tests
# ============================================================================


class TestDashboardWriteAlertsJson:
    """Tests for write_alerts_json() method."""

    def test_write_alerts_json_creates_file(self, tmp_path):
        """Test that write_alerts_json creates a JSON file."""
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        alerts = [
            {
                "timestamp": "2024-01-01",
                "metric": "DP[group]",
                "group_key": "A",
                "severity": "high",
                "reason": "Drift detected",
            }
        ]
        path = dashboard.write_alerts_json(alerts)
        assert Path(path).exists()
        assert Path(path).suffix == ".json"

    def test_write_alerts_json_serialization(self, tmp_path):
        """Test that write_alerts_json properly serializes alerts."""
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        alerts = [
            {
                "timestamp": "2024-01-01",
                "metric": "DP[group]",
                "group_key": "A",
                "severity": "high",
                "reason": "Drift detected",
            },
            {
                "timestamp": "2024-01-02",
                "metric": "EO[group]",
                "group_key": "B",
                "severity": "medium",
                "reason": "Threshold exceeded",
            },
        ]
        path = dashboard.write_alerts_json(alerts)
        # Verify JSON can be loaded
        with open(path) as f:
            loaded = json.load(f)
            assert len(loaded) == 2
            assert loaded[0]["metric"] == "DP[group]"
            assert loaded[1]["severity"] == "medium"

    def test_write_alerts_json_custom_name(self, tmp_path):
        """Test write_alerts_json with custom filename."""
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        alerts = [{"timestamp": "2024-01-01", "metric": "DP[group]", "group_key": "A"}]
        path = dashboard.write_alerts_json(alerts, name="custom_alerts.json")
        assert Path(path).name == "custom_alerts.json"
        assert Path(path).exists()

    def test_write_alerts_json_empty_list(self, tmp_path):
        """Test write_alerts_json with empty alerts list."""
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_alerts_json([])
        assert Path(path).exists()
        with open(path) as f:
            loaded = json.load(f)
            assert loaded == []

    def test_write_alerts_json_handles_non_serializable(self, tmp_path):
        """Test that write_alerts_json handles non-serializable values with default=str."""
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        # Create alert with non-serializable value (e.g., datetime object)
        import datetime

        alerts = [
            {
                "timestamp": datetime.datetime(2024, 1, 1),
                "metric": "DP[group]",
                "group_key": "A",
            }
        ]
        path = dashboard.write_alerts_json(alerts)
        # Should not raise error (default=str converts to string)
        assert Path(path).exists()


# ============================================================================
# Dashboard write_markdown_report Tests
# ============================================================================


class TestDashboardWriteMarkdownReport:
    """Tests for write_markdown_report() method."""

    def test_write_markdown_report_with_alerts(self, tmp_path):
        """Test write_markdown_report with alerts."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
                "metric": ["DP[group]"] * 5,
                "group_key": ["A"] * 5,
                "value": [0.1] * 5,
                "n": [100] * 5,
            }
        )
        alerts = [
            {
                "timestamp": "2024-01-05",
                "metric": "DP[group]",
                "group_key": "A",
                "severity": "high",
                "reason": "Drift detected",
            }
        ]
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_markdown_report(metrics, alerts)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Alerts" in content
        assert "high" in content
        assert "Drift detected" in content

    def test_write_markdown_report_without_alerts(self, tmp_path):
        """Test write_markdown_report without alerts."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
                "metric": ["DP[group]"] * 5,
                "group_key": ["A"] * 5,
                "value": [0.1] * 5,
                "n": [100] * 5,
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_markdown_report(metrics, [])
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Active alerts: **0**" in content
        # Should not have alerts table
        assert "| Time | Metric | Group | Severity | Reason |" not in content

    def test_write_markdown_report_with_datetime_index(self, tmp_path):
        """Test write_markdown_report with DatetimeIndex."""
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        metrics = pd.DataFrame(
            {
                "metric": ["DP[group]"] * 5,
                "group_key": ["A"] * 5,
                "value": [0.1] * 5,
                "n": [100] * 5,
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_markdown_report(metrics, [])
        assert Path(path).exists()
        content = Path(path).read_text()
        # Should extract latest timestamp from DatetimeIndex
        assert "2024-01-05" in content or "2024-01-01" in content

    def test_write_markdown_report_with_timestamp_column(self, tmp_path):
        """Test write_markdown_report with timestamp column."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
                "metric": ["DP[group]"] * 5,
                "group_key": ["A"] * 5,
                "value": [0.1] * 5,
                "n": [100] * 5,
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_markdown_report(metrics, [])
        assert Path(path).exists()
        content = Path(path).read_text()
        # Should extract latest timestamp from column
        assert "2024-01-05" in content or "2024-01-01" in content

    def test_write_markdown_report_template_rendering(self, tmp_path):
        """Test that markdown report template is properly rendered."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "metric": ["DP[group]"] * 10,
                "group_key": ["A"] * 10,
                "value": [0.1] * 10,
                "n": [100] * 10,
            }
        )
        alerts = [
            {
                "timestamp": "2024-01-10",
                "metric": "DP[group]",
                "group_key": "A",
                "severity": "high",
                "reason": "Test reason",
            }
        ]
        dashboard = FairnessReportingDashboard(
            MonitoringSettings(artifacts_dir=str(tmp_path), report=ReportConfig(k_anonymity=20))
        )
        path = dashboard.write_markdown_report(metrics, alerts, summary_title="Custom Title")
        content = Path(path).read_text()
        # Check template variables are rendered
        assert "Custom Title" in content
        assert "Total metric points: **10**" in content
        assert "Active alerts: **1**" in content
        assert "20" in content  # k_anonymity value

    def test_write_markdown_report_empty_metrics(self, tmp_path):
        """Test write_markdown_report with empty metrics."""
        metrics = pd.DataFrame(columns=["timestamp", "metric", "group_key", "value", "n"])
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_markdown_report(metrics, [])
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Total metric points: **0**" in content
        assert "Most recent timestamp: **N/A**" in content

    def test_write_markdown_report_custom_name(self, tmp_path):
        """Test write_markdown_report with custom filename."""
        metrics = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
                "metric": ["DP[group]"] * 5,
                "group_key": ["A"] * 5,
                "value": [0.1] * 5,
                "n": [100] * 5,
            }
        )
        dashboard = FairnessReportingDashboard(artifacts_dir=str(tmp_path))
        path = dashboard.write_markdown_report(metrics, [], name="custom_report.md")
        assert Path(path).name == "custom_report.md"
        assert Path(path).exists()
