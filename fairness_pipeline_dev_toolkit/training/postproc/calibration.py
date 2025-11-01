from __future__ import annotations

from typing import Dict, Hashable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class GroupFairnessCalibrator:
    """
    Fit/apply group-specific calibrators (Platt scaling or Isotonic regression).

    Parameters
    ----------
    method : {"platt", "isotonic"}
        Calibration method per group.
    min_samples : int
        Minimum samples required to calibrate a group (avoid overfitting).
    """

    def __init__(self, method: str = "platt", min_samples: int = 20):
        if method not in {"platt", "isotonic"}:
            raise ValueError("method must be one of {'platt','isotonic'}")
        self.method = method
        self.min_samples = int(min_samples)
        self.calibrators: Dict[Hashable, object] = {}

    def fit(
        self, scores: np.ndarray, labels: np.ndarray, groups: np.ndarray
    ) -> "GroupFairnessCalibrator":
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        groups = np.asarray(groups).reshape(-1)

        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() < self.min_samples:
                # Skip tiny groups; they fall back to raw scores at transform time
                continue
            s_g = scores[mask]
            y_g = labels[mask]
            if self.method == "platt":
                cal = LogisticRegression(max_iter=1000)
                cal.fit(s_g.reshape(-1, 1), y_g)
            else:  # isotonic
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(s_g, y_g)
            self.calibrators[g] = cal
        return self

    def transform(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        out = scores.copy()

        for g, cal in self.calibrators.items():
            mask = groups == g
            if not np.any(mask):
                continue
            s_g = scores[mask]
            if isinstance(cal, LogisticRegression):
                out[mask] = cal.predict_proba(s_g.reshape(-1, 1))[:, 1]
            elif isinstance(cal, IsotonicRegression):
                out[mask] = cal.transform(s_g)
        return out
