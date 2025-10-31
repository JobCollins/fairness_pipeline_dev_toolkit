from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pandas as pd


@dataclass
class DisparityResult:
    metrics: Dict[str, Any]  # e.g., {"approved_rate_gap": {...}}
    notes: Optional[str] = None


class DisparityDetector:
    """
    Scafold stub:
    - API only. Later add tests for proportion/mean differences with CIs.
    """

    def run(
        self, df: pd.DataFrame, *, target: Optional[str], sensitive: Sequence[str]
    ) -> DisparityResult:
        _ = (df, target, sensitive)
        return DisparityResult(metrics={}, notes="Scafold stub")
