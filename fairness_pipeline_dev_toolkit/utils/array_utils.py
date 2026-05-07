from __future__ import annotations

import numpy as np
import pandas as pd


def to_numpy_1d(x, name: str = "array") -> np.ndarray:
    """Convert array-like input to a 1-D numpy array.

    Accepts np.ndarray (passthrough), pd.Series (.to_numpy()), and list
    (np.asarray()). Raises TypeError for any other type and ValueError if the
    resulting array is not 1-D.
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    elif isinstance(x, list):
        arr = np.asarray(x)
    else:
        raise TypeError(f"{name} must be np.ndarray, pd.Series, or list. Got {type(x)}.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional. Got shape {arr.shape}.")
    return arr
