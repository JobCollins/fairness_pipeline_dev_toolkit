import numpy as np


def risk_ratio(p1: float, p2: float):
    if p1 == 0 or p2 == 0:
        return np.nan
    return float(p1 / p2)