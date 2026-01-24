from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from fairness_pipeline_dev_toolkit.utils.logging import get_logger

from .config import DriftConfig, MonitoringSettings

try:
    import pywt  # optional multi-scale

    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

logger = get_logger("monitoring.drift")


@dataclass
class AlertEvent:
    timestamp: pd.Timestamp
    metric: str
    group_key: str
    drift_score: float
    severity: str
    reason: str


class FairnessDriftAndAlertEngine:
    """
    Detects drift using KS-tests on recent vs. reference windows (over metric values),
    optionally with wavelet-based decomposition, and emits prioritized alerts.
    """

    def __init__(self, cfg: Union[DriftConfig, MonitoringSettings]):
        if isinstance(cfg, MonitoringSettings):
            self.settings = cfg
            self.cfg = cfg.drift
        else:
            self.settings = MonitoringSettings(drift=cfg)
            self.cfg = self.settings.drift
        self.events: List[AlertEvent] = []

    def _ks_drift(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
        if len(ref) < 8 or len(cur) < 8:
            return 0.0, 1.0
        stat, p = stats.ks_2samp(ref, cur, alternative="two-sided", mode="auto")
        score = float(stat * (1 - p))
        return score, float(p)

    def _decompose(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.cfg.multi_scale or not _HAS_PYWT or len(series) < 32:
            return {"full": series}
        # db2 small wavelet; limit levels to avoid over-fragmentation
        wavelet = "db2"
        max_lvl = min(5, int(np.log2(len(series))) - 1) if len(series) > 8 else 1
        coeffs = pywt.wavedec(series, wavelet, level=max_lvl)
        recon = {
            "approx": pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[
                : len(series)
            ]
        }
        for i, c in enumerate(coeffs[1:], start=1):
            zeros = [np.zeros_like(cc) for cc in coeffs]
            zeros[i] = c
            recon[f"detail_L{i}"] = pywt.waverec(zeros, wavelet)[: len(series)]
        return recon

    def analyze(
        self,
        metrics_ts: pd.DataFrame,
        window_points: int = 12,
        ref_points: int = 48,
    ) -> List[AlertEvent]:
        """
        metrics_ts: tidy frame with DatetimeIndex and columns [metric, group_key, value, n]
        Supports backward compatibility with timestamp column format.
        """
        if metrics_ts.empty:
            return []

        metrics_ts = metrics_ts.copy()
        # Handle DatetimeIndex: if timestamp is the index, reset it to a column for processing
        # Otherwise, handle as column (backward compatibility)
        if isinstance(metrics_ts.index, pd.DatetimeIndex) and metrics_ts.index.name == "timestamp":
            metrics_ts = metrics_ts.reset_index()
            metrics_ts["timestamp"] = pd.to_datetime(metrics_ts["timestamp"])
        elif "timestamp" in metrics_ts.columns:
            metrics_ts["timestamp"] = pd.to_datetime(metrics_ts["timestamp"])
        else:
            # If no timestamp column/index, try to infer from index
            if isinstance(metrics_ts.index, pd.DatetimeIndex):
                metrics_ts = metrics_ts.reset_index()
                metrics_ts["timestamp"] = pd.to_datetime(metrics_ts["timestamp"])
            else:
                raise ValueError("metrics_ts must have timestamp as DatetimeIndex or column")

        alerts: List[AlertEvent] = []

        for (metric, group_key), sub in metrics_ts.groupby(["metric", "group_key"]):
            sub = sub.sort_values("timestamp")
            vals = sub["value"].astype(float).to_numpy()
            ts = sub["timestamp"].to_numpy()
            # Extract group size (n) - use mean of recent window for severity scoring
            n_vals = (
                sub["n"].astype(float).to_numpy()
                if "n" in sub.columns
                else np.array([100] * len(sub))
            )
            n_recent = (
                int(np.nanmean(n_vals[-window_points:]))
                if len(n_vals) >= window_points
                else int(np.nanmean(n_vals)) if len(n_vals) > 0 else 100
            )

            if len(vals) < (ref_points + window_points + 2):
                continue

            # ref = vals[-(ref_points + window_points) : -window_points]
            cur = vals[-window_points:]

            # multi-scale drift
            max_score = 0.0
            pmin = 1.0
            for name, series in self._decompose(vals).items():
                ref_s = series[-(ref_points + window_points) : -window_points]
                cur_s = series[-window_points:]
                score, p = self._ks_drift(ref_s, cur_s)
                max_score = max(max_score, score)
                pmin = min(pmin, p)

            # severity scoring - now includes group size
            mag = np.nanmean(cur) if np.isfinite(cur).any() else 0.0
            sev = self._severity(metric, mag, max_score, n_recent)

            # persistence check (simple: last k points beyond thresholds)
            persistent = False
            if metric.startswith("DP"):
                persistent = np.nanmean(cur) > self.cfg.critical_dpd
            elif metric.startswith("EO"):
                persistent = np.nanmean(cur) > self.cfg.critical_eod

            # require some persistence to avoid flapping
            if persistent:
                ev = AlertEvent(
                    timestamp=pd.to_datetime(ts[-1]),
                    metric=str(metric),
                    group_key=str(group_key),
                    drift_score=float(max_score),
                    severity=sev,
                    reason=f"KS p={pmin:.4f}; mean={np.nanmean(cur):.3f}",
                )
                alerts.append(ev)

        # keep for inspection and CSV export (caller can persist)
        self.events.extend(alerts)
        return alerts

    def _severity(self, metric: str, magnitude: float, drift_score: float, group_size: int) -> str:
        """
        Compute severity score incorporating group size.
        Smaller groups reduce confidence, which may affect severity assessment.
        """
        base = self.cfg.severity_weights.get("EO" if metric.startswith("EO") else "DP", 1.0)

        # Confidence factor based on group size: larger groups = higher confidence
        # Use a sigmoid-like function: confidence increases with n, but plateaus
        # For n < 30: low confidence (penalize), n >= 100: full confidence
        min_n = 30
        optimal_n = 100
        if group_size < min_n:
            # Penalize small groups: reduce confidence significantly
            confidence_factor = max(0.3, group_size / min_n * 0.7)
        elif group_size < optimal_n:
            # Gradual increase in confidence
            confidence_factor = 0.7 + 0.3 * (group_size - min_n) / (optimal_n - min_n)
        else:
            # Full confidence for large groups
            confidence_factor = 1.0

        # Base score from magnitude and drift
        base_score = base * (0.6 * magnitude + 0.4 * drift_score)
        # Apply confidence factor: smaller groups reduce severity to avoid false alarms
        adjusted_score = base_score * confidence_factor

        if adjusted_score > 0.35:
            return "CRITICAL"
        if adjusted_score > 0.20:
            return "HIGH"
        return "LOW"
