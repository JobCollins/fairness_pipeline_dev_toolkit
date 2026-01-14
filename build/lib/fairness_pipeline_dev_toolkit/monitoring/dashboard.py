from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Union

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from .config import MonitoringSettings, ReportConfig


class FairnessReportingDashboard:
    """
    Plotly-based visualizations + Markdown reporting.
    """

    def __init__(
        self,
        settings: Union[MonitoringSettings, ReportConfig, None] = None,
        *,
        artifacts_dir: Optional[str] = None,
    ):
        if settings is None:
            base = MonitoringSettings()
        elif isinstance(settings, ReportConfig):
            base = MonitoringSettings(report=settings)
        elif isinstance(settings, MonitoringSettings):
            base = settings.with_overrides()
        else:
            raise TypeError("Unsupported monitoring settings type")

        if artifacts_dir:
            base = base.with_overrides(artifacts_dir=artifacts_dir)

        self.settings = base
        self.cfg = base.report
        os.makedirs(self.settings.artifacts_dir, exist_ok=True)
        self.settings.dump()

    def plot_trend(
        self,
        metrics_ts: pd.DataFrame,
        metric_prefix: str,
        groups: Optional[Sequence[str]] = None,
    ) -> go.Figure:
        """
        Line chart over time for a chosen fairness metric (e.g., "DP[gender]" or "EO[race]").
        """
        df = metrics_ts.copy()
        # Handle DatetimeIndex: if timestamp is the index, reset it to a column
        # Otherwise, handle as column (backward compatibility)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp":
            df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            # If no timestamp column/index, try to infer from index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise ValueError("metrics_ts must have timestamp as DatetimeIndex or column")

        df = df[df["metric"].str.startswith(metric_prefix)]
        if groups:
            df = df[df["group_key"].isin(groups)]
        fig = go.Figure()
        for gk, sub in df.groupby("group_key"):
            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=sub["value"],
                    mode="lines+markers",
                    name=str(gk),
                    hovertemplate="time=%{x}<br>value=%{y:.3f}<extra>" + str(gk) + "</extra>",
                )
            )
        fig.update_layout(
            title=f"Trend: {metric_prefix}",
            xaxis_title="Time",
            yaxis_title="Metric value",
            template="plotly_white",
            legend_title="Group",
        )
        return fig

    def plot_intersectional(
        self, metrics_ts: pd.DataFrame, metric_prefix: str, latest_only: bool = True
    ) -> go.Figure:
        """
        Heatmap visualization across intersectional subgroups.
        We show the latest timestamp per (metric, group_key), with k-anonymity suppression.
        """
        df = metrics_ts.copy()
        # Handle DatetimeIndex: if timestamp is the index, reset it to a column
        # Otherwise, handle as column (backward compatibility)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp":
            df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            # If no timestamp column/index, try to infer from index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise ValueError("metrics_ts must have timestamp as DatetimeIndex or column")

        df = df[df["metric"].str.startswith(metric_prefix)]
        if latest_only:
            idx = df.groupby(["metric", "group_key"])["timestamp"].idxmax()
            df = df.loc[idx]

        # suppress small groups
        df = df[df["n"] >= self.cfg.k_anonymity]

        if df.empty:
            # Return empty figure if no data
            return go.Figure()

        # Create pivot table for heatmap: metric as rows, group_key as columns
        pivot_df = df.pivot_table(
            index="metric", columns="group_key", values="value", aggfunc="first"
        )

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale="RdYlBu_r",  # Diverging colormap: red (high disparity) to blue (low disparity)
                text=pivot_df.values.round(3),
                texttemplate="%{text:.3f}",
                textfont={"size": 10},
                colorbar=dict(title="Metric Value"),
                hovertemplate="Metric=%{y}<br>Group=%{x}<br>Value=%{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Intersectional snapshot: {metric_prefix}",
            xaxis_title="Intersectional Group",
            yaxis_title="Metric",
            template="plotly_white",
            height=max(400, len(pivot_df.index) * 40),  # Adjust height based on number of metrics
        )
        return fig

    def write_alerts_json(self, alerts: List[dict], name: str = "active_alerts.json") -> str:
        path = os.path.join(self.settings.artifacts_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, default=str)
        return path

    def write_markdown_report(
        self,
        metrics_ts: pd.DataFrame,
        alerts: List[dict],
        name: str = "report.md",
        summary_title: str = "Fairness Monitoring Report",
    ) -> str:
        """
        Simple, human-readable Markdown report.
        """
        tpl = Template(
            """# {{ title }}

_This report summarizes fairness metrics, recent drift, and active alerts._

## Summary
- Total metric points: **{{ n_points }}**
- Most recent timestamp: **{{ latest }}**
- Active alerts: **{{ n_alerts }}**

{% if alerts %}
## Alerts
| Time | Metric | Group | Severity | Reason |
|---|---|---|---|---|
{% for a in alerts %}
| {{ a['timestamp'] }} | {{ a['metric'] }} | {{ a['group_key'] }} | **{{ a['severity'] }}** | {{ a['reason'] }} |
{% endfor %}
{% endif %}

## Notes
- Metrics with group size < {{ k }} are suppressed.
- DP threshold: difference > 0.10 flagged; EO threshold: > 0.10 flagged.
"""
        )
        # Handle DatetimeIndex for latest timestamp extraction
        if metrics_ts.empty:
            latest = "N/A"
        elif (
            isinstance(metrics_ts.index, pd.DatetimeIndex) and metrics_ts.index.name == "timestamp"
        ):
            latest = str(metrics_ts.index.max())
        elif "timestamp" in metrics_ts.columns:
            latest = str(metrics_ts["timestamp"].max())
        elif isinstance(metrics_ts.index, pd.DatetimeIndex):
            latest = str(metrics_ts.index.max())
        else:
            latest = "N/A"
        md = tpl.render(
            title=summary_title,
            n_points=len(metrics_ts),
            latest=str(latest),
            n_alerts=len(alerts),
            alerts=alerts,
            k=self.cfg.k_anonymity,
        )
        path = os.path.join(self.settings.artifacts_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        return path
