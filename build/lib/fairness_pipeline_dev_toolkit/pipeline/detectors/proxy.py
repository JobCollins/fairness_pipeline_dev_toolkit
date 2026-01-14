from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd


@dataclass
class ProxyResult:
    associations: Dict[str, Dict[str, float]]  # {feature: {attr: association_score}}
    notes: Optional[str] = None


class ProxyDetector:
    """
    Scafold stub:
    - API only. Later compute Pearson/Cramér's V/MI by dtype.
    """

    def run(
        self, df: pd.DataFrame, *, features: Sequence[str], sensitive: Sequence[str]
    ) -> ProxyResult:
        _ = (df, features, sensitive)
        return ProxyResult(associations={}, notes="Scafold stub")
