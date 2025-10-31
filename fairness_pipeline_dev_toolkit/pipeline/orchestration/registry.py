from __future__ import annotations

from typing import Dict, Type

from ..transformers import (
    DisparateImpactRemover,
    InstanceReweighting,
    ProxyDropper,
    ReweighingTransformer,
)

# from sklearn.base import BaseEstimator


# Map short names used in config -> classes
# NOTE: keys are *stable* transformer names used in YAML config
_TRANSFORMERS: Dict[str, Type] = {
    "InstanceReweighting": InstanceReweighting,
    "DisparateImpactRemover": DisparateImpactRemover,
    "ReweighingTransformer": ReweighingTransformer,
    "ProxyDropper": ProxyDropper,
}


def get_transformer_class(name: str):
    try:
        return _TRANSFORMERS[name]
    except KeyError:
        raise ValueError(f"Unknown transformer '{name}'. Available: {list(_TRANSFORMERS.keys())}")
