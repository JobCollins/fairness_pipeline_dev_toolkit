from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd

@dataclass
class InputSpec:
    y_true: Optional[Iterable] = None
    y_pred: Optional[Iterable] = None
    scores: Optional[Iterable] = None
    attrs_df: Optional[pd.DataFrame] = None

def coerce_arrays(y_true=None, y_pred=None, scores=None):
    y_true = None if y_true is None else np.asarray(y_true)
    y_pred = None if y_pred is None else np.asarray(y_pred)
    scores = None if scores is None else np.asarray(scores)
    return y_true, y_pred, scores

def check_lengths(*arrays):
    lengths = [len(a) for a in arrays if a is not None]
    if lengths and len(set(lengths)) != 1:
        raise ValueError(f"Mismatched lengths: {lengths}")