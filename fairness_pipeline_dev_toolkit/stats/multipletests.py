from __future__ import annotations

from typing import Tuple

import numpy as np


def bonferroni(pvals):
    """
    Bonferroni correction: p_adj = min(1, p * n_tests)
    Returns adjusted p-values. Same shape as input.
    """
    pvals = np.asarray(pvals, dtype=float)
    n_tests = pvals.size
    return np.minimum(1.0, pvals * n_tests)


def benjamini_hochberg(pvals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg correction for controlling the false discovery rate.
    Returns
    -------
    adjusted_sorted : np.ndarray
        BH-adjusted p-values in **ascending-sorted p-value order** (not original input order).
    sorted_idx : np.ndarray
        Indices that sort the input p-values ascending; align with ``adjusted_sorted`` entry-wise.
    """
    pvals = np.asarray(pvals, dtype=float)
    n_tests = pvals.size
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    # pvals_i * n_tests / i
    adjusted_pvals = sorted_pvals * n_tests / (np.arange(1, n_tests + 1))
    # Cumulative minimum to ensure monotonicity
    adjusted_pvals = np.minimum.accumulate(adjusted_pvals[::-1])[::-1]
    # Cap at 1.0
    adjusted_pvals = np.clip(adjusted_pvals, 0, 1)
    return adjusted_pvals, sorted_idx
