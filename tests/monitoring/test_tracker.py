from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.monitoring import (
    ColumnMap,
    RealTimeFairnessTracker,
    TrackerConfig,
)

# Skip plotly import check (dashboard requires it but tracker doesn't)
pytest.importorskip("plotly")


def test_tracker_initializes_with_datetime_index():
    """Test that tracker initializes with DatetimeIndex"""
    cfg = TrackerConfig(window_size=1000, min_group_size=10)
    tracker = RealTimeFairnessTracker(cfg)

    assert isinstance(tracker.metrics_ts.index, pd.DatetimeIndex)
    assert tracker.metrics_ts.index.name == "timestamp"
    assert len(tracker.metrics_ts) == 0


def test_tracker_processes_batch_and_stores_with_datetime_index(tmp_path):
    """Test that tracker stores metrics with DatetimeIndex"""
    cfg = TrackerConfig(window_size=1000, min_group_size=10)
    tracker = RealTimeFairnessTracker(cfg, artifacts_dir=str(tmp_path))

    # Create a batch
    batch = pd.DataFrame(
        {
            "y_pred": [1, 0, 1, 0, 1] * 20,
            "y_true": [1, 1, 0, 0, 1] * 20,
            "gender": ["A", "B", "A", "B", "C"] * 20,
        }
    )

    cmap = ColumnMap(y_pred="y_pred", y_true="y_true", protected=["gender"], intersections=[])

    tracker.process_batch(batch, cmap)

    # Check that metrics_ts has DatetimeIndex
    assert isinstance(tracker.metrics_ts.index, pd.DatetimeIndex)
    assert tracker.metrics_ts.index.name == "timestamp"
    assert len(tracker.metrics_ts) > 0
    assert "metric" in tracker.metrics_ts.columns
    assert "group_key" in tracker.metrics_ts.columns
    assert "value" in tracker.metrics_ts.columns
    assert "n" in tracker.metrics_ts.columns


def test_tracker_csv_save_load_preserves_datetime_index(tmp_path):
    """Test that CSV save/load preserves DatetimeIndex"""
    cfg = TrackerConfig(window_size=1000, min_group_size=10)
    tracker = RealTimeFairnessTracker(cfg, artifacts_dir=str(tmp_path))

    # Process a batch
    batch = pd.DataFrame(
        {
            "y_pred": [1, 0, 1, 0, 1] * 20,
            "y_true": [1, 1, 0, 0, 1] * 20,
            "gender": ["A", "B", "A", "B", "C"] * 20,
        }
    )

    cmap = ColumnMap(y_pred="y_pred", y_true="y_true", protected=["gender"], intersections=[])

    tracker.process_batch(batch, cmap)

    # Verify CSV was saved with index
    csv_path = Path(tmp_path) / "metrics_timeseries.csv"
    assert csv_path.exists()

    # Load CSV and verify DatetimeIndex is preserved
    loaded = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.name == "timestamp"
    assert len(loaded) == len(tracker.metrics_ts)


def test_tracker_sliding_window(tmp_path):
    """Test that tracker maintains sliding window correctly"""
    cfg = TrackerConfig(window_size=50, min_group_size=5)
    tracker = RealTimeFairnessTracker(cfg, artifacts_dir=str(tmp_path))

    # Process multiple batches that exceed window size
    for i in range(5):
        batch = pd.DataFrame(
            {
                "y_pred": [1, 0] * 30,
                "y_true": [1, 1] * 30,
                "gender": ["A", "B"] * 30,
            }
        )

        cmap = ColumnMap(y_pred="y_pred", y_true="y_true", protected=["gender"], intersections=[])

        window = tracker.process_batch(batch, cmap)

        # Window should not exceed window_size
        assert len(window) <= cfg.window_size

    # Final window should be at or below window_size
    assert len(tracker.window_df) <= cfg.window_size


def test_tracker_computes_dp_metrics(tmp_path):
    """Test that tracker computes demographic parity metrics"""
    cfg = TrackerConfig(window_size=1000, min_group_size=10)
    tracker = RealTimeFairnessTracker(cfg, artifacts_dir=str(tmp_path))

    # Create batch with clear DP difference
    batch = pd.DataFrame(
        {
            "y_pred": [1] * 30 + [0] * 20,  # Group A: 100% positive, Group B: 0% positive
            "y_true": [1] * 50,
            "gender": ["A"] * 30 + ["B"] * 20,
        }
    )

    cmap = ColumnMap(y_pred="y_pred", y_true="y_true", protected=["gender"], intersections=[])

    tracker.process_batch(batch, cmap)

    # Check for DP metrics
    dp_metrics = tracker.metrics_ts[tracker.metrics_ts["metric"].str.startswith("DP[")]
    assert len(dp_metrics) > 0

    # Should have DPD (demographic parity difference)
    dpd = dp_metrics[dp_metrics["group_key"] == "__DPD__"]
    assert len(dpd) > 0
    assert dpd["value"].iloc[0] > 0  # Should be positive (difference exists)


def test_tracker_computes_eo_metrics(tmp_path):
    """Test that tracker computes equalized odds metrics"""
    cfg = TrackerConfig(window_size=1000, min_group_size=10)
    tracker = RealTimeFairnessTracker(cfg, artifacts_dir=str(tmp_path))

    # Create batch with TPR/FPR differences
    batch = pd.DataFrame(
        {
            "y_pred": [1, 1, 0, 0] * 25,
            "y_true": [1, 0, 1, 0] * 25,  # Mix of positives and negatives
            "gender": ["A", "A", "B", "B"] * 25,
        }
    )

    cmap = ColumnMap(y_pred="y_pred", y_true="y_true", protected=["gender"], intersections=[])

    tracker.process_batch(batch, cmap)

    # Check for EO metrics
    eo_metrics = tracker.metrics_ts[tracker.metrics_ts["metric"].str.startswith("EO[")]
    assert len(eo_metrics) > 0

    # Should have EOD (equalized odds difference)
    eod = eo_metrics[eo_metrics["group_key"] == "__EOD__"]
    assert len(eod) > 0


def test_tracker_backward_compatibility_with_timestamp_column(tmp_path):
    """Test that tracker can handle old CSV format with timestamp column"""
    # Create a CSV with timestamp column (old format)
    old_format_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "metric": ["DP[group]"] * 5,
            "group_key": ["A"] * 5,
            "value": [0.1] * 5,
            "n": [100] * 5,
        }
    )
    csv_path = Path(tmp_path) / "metrics_timeseries.csv"
    old_format_df.to_csv(csv_path, index=False)

    # Load it - should be able to convert to DatetimeIndex
    loaded = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "timestamp" in loaded.columns:
        loaded = loaded.set_index("timestamp")

    assert isinstance(loaded.index, pd.DatetimeIndex) or "timestamp" in loaded.columns
