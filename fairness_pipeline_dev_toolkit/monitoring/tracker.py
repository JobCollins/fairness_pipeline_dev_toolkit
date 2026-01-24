from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.utils.logging import get_logger

logger = get_logger("monitoring.tracker")


@dataclass
class ColumnMap:
    """Maps dataset columns → monitoring contract."""

    y_pred: str  # predictions/probabilities (0..1) or binary
    y_true: Optional[str] = None  # optional if running score-only
    protected: Sequence[str] = ()  # e.g., ["gender","race","age_band"]
    intersections: Sequence[Sequence[str]] = ()  # e.g., [["gender","race"]]


@dataclass
class TrackerConfig:
    window_size: int = 10_000  # count-based ring buffer
    min_group_size: int = 30
    metrics: Sequence[str] = ("demographic_parity", "equalized_odds")  # DP, EO


def _binarize(p: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    if p.ndim != 1:
        p = p.ravel()
    return (p >= thresh).astype(int)


def _group_keys(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    if not keys:  # overall
        return pd.Series(["__overall__"] * len(df), index=df.index)
    return df[keys].astype(str).agg("×".join, axis=1)


class RealTimeFairnessTracker:
    """
    Ingests batches and computes fairness metrics over a sliding window.
    Stores tidy time-series outputs for dashboarding & drift detection.
    """

    def __init__(self, cfg: TrackerConfig, artifacts_dir: str = "artifacts/monitoring"):
        self.cfg = cfg
        self.artifacts_dir = artifacts_dir
        self._buffer: deque[pd.DataFrame] = deque()
        self._n: int = 0
        # Initialize with DatetimeIndex as required
        self.metrics_ts = pd.DataFrame(
            columns=["metric", "group_key", "value", "n"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )

    def _append_to_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maintain a ring buffer with at most window_size rows."""
        if df.empty:
            return self.window_df
        self._buffer.append(df)
        self._n += len(df)
        # Trim from the left until within budget
        while self._n > self.cfg.window_size and self._buffer:
            left = self._buffer[0]
            overflow = self._n - self.cfg.window_size
            if len(left) <= overflow:
                self._buffer.popleft()
                self._n -= len(left)
            else:
                # split left chunk
                keep = left.iloc[len(left) - (len(left) - overflow) :]
                self._buffer[0] = keep
                self._n = self.cfg.window_size
        if not self._buffer:
            return pd.DataFrame()
        return pd.concat(list(self._buffer), ignore_index=True)

    @property
    def window_df(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame()
        return pd.concat(list(self._buffer), ignore_index=True)

    def _demographic_parity(self, yhat: np.ndarray, group: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"yhat": yhat, "g": group})
        grp = df.groupby("g")["yhat"].mean()
        if grp.empty:
            return pd.DataFrame(columns=["group_key", "value", "n"])
        pr = grp.max()
        un = grp.min()
        # We log both rate per group and the disparity summary as separate rows
        rows = []
        counts = df.groupby("g").size().to_dict()
        for k, v in grp.to_dict().items():
            rows.append({"group_key": str(k), "value": float(v), "n": int(counts.get(k, 0))})
        rows.append({"group_key": "__DPD__", "value": float(pr - un), "n": int(len(df))})
        return pd.DataFrame(rows)

    def _equalized_odds(self, yhat: np.ndarray, y: np.ndarray, group: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"yhat": yhat, "y": y, "g": group})

        # TPR/FPR by group, then EO as max diff across TPR and FPR
        def rates(sub):
            positives = sub["y"] == 1
            negatives = sub["y"] == 0
            tpr = (sub.loc[positives, "yhat"] == 1).mean() if positives.any() else np.nan
            fpr = (sub.loc[negatives, "yhat"] == 1).mean() if negatives.any() else np.nan
            return tpr, fpr

        res = {}
        counts = {}
        for gk, sub in df.groupby("g"):
            res[gk] = rates(sub)
            counts[gk] = len(sub)

        # per-group echoes (tpr/fpr) + EO summary
        rows = []
        tprs = [v[0] for v in res.values() if pd.notna(v[0])]
        fprs = [v[1] for v in res.values() if pd.notna(v[1])]
        eo = 0.0
        if tprs:
            eo = max(eo, float(np.nanmax(tprs) - np.nanmin(tprs)))
        if fprs:
            eo = max(eo, float(np.nanmax(fprs) - np.nanmin(fprs)))
        for gk, (tpr, fpr) in res.items():
            rows.append(
                {
                    "group_key": f"{gk}::TPR",
                    "value": float(tpr) if pd.notna(tpr) else np.nan,
                    "n": counts[gk],
                }
            )
            rows.append(
                {
                    "group_key": f"{gk}::FPR",
                    "value": float(fpr) if pd.notna(fpr) else np.nan,
                    "n": counts[gk],
                }
            )
        rows.append({"group_key": "__EOD__", "value": eo, "n": int(len(df))})
        return pd.DataFrame(rows)

    def _emit_metric_rows(self, ts: pd.Timestamp, metric: str, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        out = rows.copy()
        out.insert(0, "metric", metric)
        # Set timestamp as index for each row
        out.index = [ts] * len(out)
        out.index.name = "timestamp"
        self.metrics_ts = pd.concat([self.metrics_ts, out])
        # persist incrementally - preserve DatetimeIndex
        path = f"{self.artifacts_dir}/metrics_timeseries.csv"
        self.metrics_ts.to_csv(path, index=True)

    def process_batch(self, batch: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
        """
        Ingest a new batch, recompute metrics on the sliding window, and
        append rows to the time series store. Returns the current window df.
        """
        logger.debug(
            "Processing batch",
            extra={"batch_size": len(batch), "window_size": self.cfg.window_size},
        )
        ts = pd.Timestamp.utcfromtimestamp(time.time())
        df = batch.copy()
        if cmap.y_pred not in df.columns:
            raise KeyError(f"Missing predictions column: {cmap.y_pred}")
        if cmap.y_true and cmap.y_true not in df.columns:
            raise KeyError(f"Missing labels column: {cmap.y_true}")
        # normalize predictions into probabilities if needed (assume already 0..1)
        # enforce protected availability (we’ll compute per-protected and intersections)
        for col in cmap.protected:
            if col not in df.columns:
                raise KeyError(f"Missing protected attribute column: {col}")

        window = self._append_to_window(df)
        if window.empty:
            return window

        yhat_prob = window[cmap.y_pred].to_numpy()
        yhat = _binarize(yhat_prob)
        y = window[cmap.y_true].to_numpy() if cmap.y_true else None

        # 1) single-attribute groups
        for attr in cmap.protected:
            g = window[attr].astype(str).to_numpy()
            # DP
            if "demographic_parity" in self.cfg.metrics:
                rows = self._demographic_parity(yhat, g)
                rows = rows[rows["n"] >= self.cfg.min_group_size]
                self._emit_metric_rows(ts, f"DP[{attr}]", rows)
            # EO
            if y is not None and "equalized_odds" in self.cfg.metrics:
                rows = self._equalized_odds(yhat, y, g)
                rows = rows[rows["n"] >= self.cfg.min_group_size]
                self._emit_metric_rows(ts, f"EO[{attr}]", rows)

        # 2) intersections
        for combo in cmap.intersections:
            if not combo:
                continue
            key = "+".join(combo)
            g = _group_keys(window, combo).to_numpy()
            if "demographic_parity" in self.cfg.metrics:
                rows = self._demographic_parity(yhat, g)
                rows = rows[rows["n"] >= self.cfg.min_group_size]
                self._emit_metric_rows(ts, f"DP[{key}]", rows)
            if y is not None and "equalized_odds" in self.cfg.metrics:
                rows = self._equalized_odds(yhat, y, g)
                rows = rows[rows["n"] >= self.cfg.min_group_size]
                self._emit_metric_rows(ts, f"EO[{key}]", rows)

        # 3) overall echoes (for quick dashboards)
        if "demographic_parity" in self.cfg.metrics:
            rows = self._demographic_parity(yhat, np.array(["__overall__"] * len(window)))
            self._emit_metric_rows(ts, "DP[overall]", rows)
        if y is not None and "equalized_odds" in self.cfg.metrics:
            rows = self._equalized_odds(yhat, y, np.array(["__overall__"] * len(window)))
            self._emit_metric_rows(ts, "EO[overall]", rows)

        return window
