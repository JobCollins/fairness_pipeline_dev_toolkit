from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd


@dataclass
class RepresentationResult:
    by_group: Dict[str, Dict[str, float]]  # {attr: {group: proportion}}
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None
    notes: Optional[str] = None


class RepresentationDetector:
    """
    Scaffold stub:
    - API only. Next we'll compute group proportions and compare to benchmarks.
    """

    def run(
        self,
        df: pd.DataFrame,
        *,
        sensitive: Sequence[str],
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> RepresentationResult:
        _ = (df, sensitive, benchmarks)
        # Phase 0: return empty structure to prove wiring
        return RepresentationResult(by_group={}, benchmarks=benchmarks, notes="Scaffold stub")
