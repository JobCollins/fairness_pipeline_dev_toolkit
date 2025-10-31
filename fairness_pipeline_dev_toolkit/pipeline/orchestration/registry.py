from __future__ import annotations

from typing import Dict, Type

from sklearn.base import BaseEstimator

from ..transformers import DisparateImpactRemover, InstanceReweighting

# Map short names used in config -> classes
TRANSFORMER_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    "disparate_impact": DisparateImpactRemover,
    "reweighting": InstanceReweighting,
}
