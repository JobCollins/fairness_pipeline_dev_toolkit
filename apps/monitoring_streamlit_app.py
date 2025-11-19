import pandas as pd
import streamlit as st

from fairness_pipeline_dev_toolkit.monitoring import (
    ColumnMap,
    FairnessDriftAndAlertEngine,
    FairnessReportingDashboard,
    MonitoringSettings,
    RealTimeFairnessTracker,
    TrackerConfig,
)

st.set_page_config(page_title="Fairness Monitoring", layout="wide")
st.title("Fairness Monitoring — Streamlit")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
art_dir = st.sidebar.text_input("Artifacts dir", "artifacts/monitoring")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data preview", df.head())

    # Column mapping
    pred_col = st.selectbox("Predictions column", options=df.columns, index=0)
    ytrue_col = st.selectbox("Labels column (optional)", options=["<none>"] + list(df.columns))
    sensitive_cols = st.multiselect("Sensitive attributes", options=df.columns)

    # intersection picker
    inter_choices = []
    if sensitive_cols:
        # allow choosing 2-way intersections quickly
        for i in range(len(sensitive_cols)):
            for j in range(i + 1, len(sensitive_cols)):
                inter_choices.append((sensitive_cols[i], sensitive_cols[j]))
    inter_selected = st.multiselect("Intersections (pairs)", options=inter_choices)

    cfg = TrackerConfig(window_size=10_000, min_group_size=30)
    tracker = RealTimeFairnessTracker(cfg, artifacts_dir=art_dir)

    if st.button("Process batch"):
        cmap = ColumnMap(
            y_pred=pred_col,
            y_true=None if ytrue_col == "<none>" else ytrue_col,
            protected=sensitive_cols,
            intersections=[list(x) for x in inter_selected],
        )
        tracker.process_batch(df, cmap)
        st.success("Batch processed. Metrics updated.")

    # Show latest metrics
    try:
        # Load CSV and set DatetimeIndex if timestamp column exists, or use index if already DatetimeIndex
        mts = pd.read_csv(f"{art_dir}/metrics_timeseries.csv", index_col=0, parse_dates=True)
        # If index is not DatetimeIndex, try to set it from timestamp column
        if not isinstance(mts.index, pd.DatetimeIndex):
            if "timestamp" in mts.columns:
                mts = mts.set_index("timestamp")
            elif mts.index.name == "timestamp":
                mts.index = pd.to_datetime(mts.index)
        # Ensure index name is "timestamp"
        mts.index.name = "timestamp"
        st.subheader("Metrics Time Series")
        st.dataframe(mts.tail(50))
        settings = MonitoringSettings(artifacts_dir=art_dir)
        # Drift
        drift = FairnessDriftAndAlertEngine(settings)
        alerts = drift.analyze(mts)
        st.subheader("Active Alerts")
        if alerts:
            st.write(pd.DataFrame([a.__dict__ for a in alerts]))
        else:
            st.info("No active alerts based on current thresholds.")

        # Plots
        dash = FairnessReportingDashboard(settings)
        with st.expander("Trend — DP[overall]"):
            st.plotly_chart(dash.plot_trend(mts, "DP[overall]"), use_container_width=True)
        if sensitive_cols:
            first = sensitive_cols[0]
            with st.expander(f"Trend — DP[{first}]"):
                st.plotly_chart(dash.plot_trend(mts, f"DP[{first}]"), use_container_width=True)
        with st.expander("Intersection snapshot (latest) — DP"):
            st.plotly_chart(dash.plot_intersectional(mts, "DP["), use_container_width=True)

        # Report
        if st.button("Write Markdown Report"):
            alerts_df = [a.__dict__ for a in alerts]
            dash.write_alerts_json(alerts_df)
            p = dash.write_markdown_report(mts, alerts_df)
            st.success(f"Report written: {p}")

    except FileNotFoundError:
        st.info("No metrics yet. Process a batch to get started.")
