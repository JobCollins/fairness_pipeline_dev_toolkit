# Minimal Dash scaffold to keep the door open for enterprise embedding.
# Launch manually if/when needed.

import os

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html

app = dash.Dash(__name__)
art_dir = os.environ.get("FPDT_ART_DIR", "artifacts/monitoring")


def _load():
    try:
        # Load CSV and set DatetimeIndex if timestamp column exists, or use index if already DatetimeIndex
        df = pd.read_csv(f"{art_dir}/metrics_timeseries.csv", index_col=0, parse_dates=True)
        # If index is not DatetimeIndex, try to set it from timestamp column
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif df.index.name == "timestamp":
                df.index = pd.to_datetime(df.index)
        # Ensure index name is "timestamp"
        df.index.name = "timestamp"
        return df
    except Exception:
        # Return empty DataFrame with DatetimeIndex
        return pd.DataFrame(
            columns=["metric", "group_key", "value", "n"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )


app.layout = html.Div(
    [
        html.H2("Fairness Monitoring (Dash)"),
        dcc.Dropdown(id="metric", options=[], placeholder="Select metric"),
        dcc.Graph(id="trend"),
        dcc.Interval(id="tick", interval=15_000, n_intervals=0),
    ]
)


@app.callback(
    dash.Output("metric", "options"),
    dash.Output("metric", "value"),
    dash.Input("tick", "n_intervals"),
)
def _ops(_):
    df = _load()
    metrics = sorted(df["metric"].unique()) if not df.empty else []
    val = metrics[0] if metrics else None
    return [{"label": m, "value": m} for m in metrics], val


@app.callback(
    dash.Output("trend", "figure"),
    dash.Input("metric", "value"),
    dash.Input("tick", "n_intervals"),
)
def _trend(metric, _):
    df = _load()
    if not metric or df.empty:
        return go.Figure()
    sub = df[df["metric"] == metric].copy()
    # Handle DatetimeIndex: if timestamp is the index, use it directly
    if isinstance(sub.index, pd.DatetimeIndex):
        # Use index as timestamps
        timestamps = sub.index
    elif "timestamp" in sub.columns:
        timestamps = pd.to_datetime(sub["timestamp"])
    else:
        timestamps = sub.index
    fig = go.Figure()
    for gk, gdf in sub.groupby("group_key"):
        # Get timestamps for this group
        if isinstance(sub.index, pd.DatetimeIndex):
            group_timestamps = gdf.index
        else:
            group_timestamps = (
                timestamps[gdf.index] if hasattr(timestamps, "__getitem__") else timestamps
            )
        fig.add_trace(
            go.Scatter(x=group_timestamps, y=gdf["value"], mode="lines+markers", name=str(gk))
        )
    fig.update_layout(template="plotly_white", title=f"Trend: {metric}")
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8055)
