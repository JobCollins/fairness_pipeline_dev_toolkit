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
        return pd.read_csv(f"{art_dir}/metrics_timeseries.csv")
    except Exception:
        return pd.DataFrame(columns=["timestamp", "metric", "group_key", "value", "n"])


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
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    fig = go.Figure()
    for gk, gdf in sub.groupby("group_key"):
        fig.add_trace(
            go.Scatter(x=gdf["timestamp"], y=gdf["value"], mode="lines+markers", name=str(gk))
        )
    fig.update_layout(template="plotly_white", title=f"Trend: {metric}")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8055)
