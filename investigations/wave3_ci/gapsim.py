"""Vectorised interval methods for the max−min gap of K binary group rates.

Every function takes a batch of S simulated datasets summarised by positive counts
``X`` (S, K) and group sizes ``n`` (K,). For binary outcomes this summary is
sufficient, and each resampling scheme has an exact count-level equivalent:

- M0 (current analyzer): resample N row indices with replacement from the pooled
  data == Multinomial(N, cell shares) over the 2K (group, label) cells. Group sizes
  vary per draw; a group with 0 draws gives NaN (as ``dpd_stat_from_indices``).
- M1 (stratified): resample n_k rows within group k == Binomial(n_k, p̂_k).
- M3 (permutation of group labels) == sequential hypergeometric split of the
  pooled positives into groups of the fixed sizes n_k.

``check_vectorised_vs_package.py`` cross-checks M0 against the real analyzer.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

ALPHA = 0.05
Z_LO, Z_HI = norm.ppf(ALPHA / 2), norm.ppf(1 - ALPHA / 2)
TOL = 1e-12


def gap(rates: np.ndarray) -> np.ndarray:
    """max − min over the last axis; NaN propagates (any missing group -> NaN)."""
    return rates.max(axis=-1) - rates.min(axis=-1)


# ----------------------------------------------------------------- replicates
def m0_replicates(rng, X, n, B) -> Tuple[np.ndarray, np.ndarray]:
    """Unstratified index bootstrap, as FairnessAnalyzer does today."""
    S, K = X.shape
    N = int(n.sum())
    cells = np.concatenate([X, n[None, :] - X], axis=1).astype(float)  # pos_k..., neg_k...
    probs = cells / N
    C = 2 * K
    draws = np.empty((S, B, C), dtype=np.int64)
    remaining = np.full((S, B), N, dtype=np.int64)
    for j in range(C - 1):
        rem_p = probs[:, j:].sum(axis=1)
        cond = np.where(rem_p > 0, np.clip(probs[:, j] / np.where(rem_p > 0, rem_p, 1), 0, 1), 0)
        x = rng.binomial(remaining, cond[:, None])
        draws[..., j] = x
        remaining -= x
    draws[..., -1] = remaining
    pos = draws[..., :K]
    tot = pos + draws[..., K:]
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = pos / tot
    return gap(rates), tot


def m1_rates(rng, X, n, B, m=None) -> np.ndarray:
    """Stratified (within-group) bootstrap rates; optional m_k-out-of-n_k."""
    S, K = X.shape
    phat = X / n[None, :]
    size = n if m is None else m
    return rng.binomial(size[None, None, :], phat[:, None, :], size=(S, B, K)) / size


# ----------------------------------------------------------------- intervals
def percentile(boot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = np.nanpercentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=1)
    return lo, hi


def _row_quantile(sorted_boot, nf, q):
    """numpy 'linear' quantile per row on the first nf (finite) sorted entries."""
    pos = q * (nf - 1)
    lo_i = np.floor(pos).astype(int)
    hi_i = np.minimum(lo_i + 1, nf - 1)
    frac = pos - lo_i
    rows = np.arange(sorted_boot.shape[0])
    a = sorted_boot[rows, lo_i]
    b = sorted_boot[rows, hi_i]
    return a + frac * (b - a)


def jackknife_accel(X, n) -> np.ndarray:
    """Exact jackknife acceleration of the gap over all N observations (package formula)."""
    S, K = X.shape
    phat = X / n[None, :]
    J = np.empty((S, 2 * K))
    W = np.empty((S, 2 * K))
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(K):
            for lab, col in ((1, k), (0, K + k)):
                r = phat.copy()
                r[:, k] = (X[:, k] - lab) / (n[k] - 1) if n[k] > 1 else np.nan
                J[:, col] = gap(r)
                W[:, col] = X[:, k] if lab == 1 else n[k] - X[:, k]
    N = n.sum()
    jm = np.sum(W * J, axis=1) / N
    d = jm[:, None] - J
    num = np.sum(W * d**3, axis=1)
    den = 6.0 * (np.sum(W * d**2, axis=1) ** 1.5 + 1e-12)
    return num / den


def bca(boot, theta, accel, *, drop_nan: bool) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """BCa exactly as stats.bootstrap.bca_ci (drop_nan=False) or with NaNs removed first.

    drop_nan=False reproduces the package: NaN replicates count as "not less" in z0
    and np.percentile returns NaN when any replicate is NaN; NaN accel -> ValueError
    in the package, reported here as NaN.
    """
    finite = np.isfinite(boot)
    nf = finite.sum(axis=1)
    less = (boot < theta[:, None]) & finite
    if drop_nan:
        p_less = less.sum(axis=1) / np.maximum(nf, 1)
    else:
        p_less = less.mean(axis=1)
    z0 = norm.ppf(np.clip(p_less, 1e-6, 1 - 1e-6))
    out = []
    for z in (Z_LO, Z_HI):
        with np.errstate(invalid="ignore", divide="ignore"):
            q = norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
        sb = np.sort(np.where(finite, boot, np.inf), axis=1)
        qq = np.where(np.isfinite(q), q, 0.5)
        v = _row_quantile(sb, np.maximum(nf, 1), qq)
        v = np.where(np.isfinite(q), v, np.nan)
        if not drop_nan:
            v = np.where(nf == boot.shape[1], v, np.nan)
        else:
            v = np.where(nf >= 2, v, np.nan)
        out.append(v)
    return out[0], out[1], {"z0": z0, "nan_share": 1 - nf / boot.shape[1]}


def pairwise_bounds(L, U) -> Tuple[np.ndarray, np.ndarray]:
    """Gap bounds from simultaneous pairwise-difference intervals (S, P)."""
    lb = np.max(np.maximum(0.0, np.maximum(L, -U)), axis=1)
    ub = np.minimum(np.max(np.maximum(np.abs(L), np.abs(U)), axis=1), 1.0)
    return lb, ub


def m2_bonferroni_ac(X, n):
    """Bonferroni Agresti–Caffo intervals for every p_i − p_j."""
    S, K = X.shape
    pairs = list(combinations(range(K), 2))
    z = norm.ppf(1 - ALPHA / (2 * len(pairs)))
    pt = (X + 1) / (n[None, :] + 2)
    L = np.empty((S, len(pairs)))
    U = np.empty_like(L)
    for c, (i, j) in enumerate(pairs):
        d = pt[:, i] - pt[:, j]
        se = np.sqrt(
            pt[:, i] * (1 - pt[:, i]) / (n[i] + 2) + pt[:, j] * (1 - pt[:, j]) / (n[j] + 2)
        )
        L[:, c] = np.clip(d - z * se, -1, 1)
        U[:, c] = np.clip(d + z * se, -1, 1)
    return pairwise_bounds(L, U)


def m2_maxt(X, n, rates_star):
    """Bootstrap max-|t| simultaneous intervals from stratified replicates (S, B, K)."""
    S, K = X.shape
    phat = X / n[None, :]
    pairs = list(combinations(range(K), 2))
    i_idx = np.array([p[0] for p in pairs])
    j_idx = np.array([p[1] for p in pairs])
    dstar = rates_star[..., i_idx] - rates_star[..., j_idx]  # (S, B, P)
    dhat = phat[:, i_idx] - phat[:, j_idx]  # (S, P)
    se = dstar.std(axis=1, ddof=1)
    safe = np.where(se > 0, se, 1.0)
    t = np.abs(dstar - dhat[:, None, :]) / safe[:, None, :]
    t = np.where(se[:, None, :] > 0, t, 0.0)
    c = np.quantile(t.max(axis=2), 1 - ALPHA, axis=1)
    L = dhat - c[:, None] * se
    U = dhat + c[:, None] * se
    return pairwise_bounds(L, U)


def m3_permutation(rng, X, n, B):
    """Permutation p-values for H0: all K rates equal (gap and chi-square statistics)."""
    S, K = X.shape
    N = int(n.sum())
    tot_pos = X.sum(axis=1)
    good = np.broadcast_to(tot_pos[:, None], (S, B)).astype(np.int64).copy()
    bad = (N - good).astype(np.int64)
    perm = np.empty((S, B, K))
    for k in range(K - 1):
        x = rng.hypergeometric(good, bad, int(n[k]))
        perm[..., k] = x
        good -= x
        bad -= int(n[k]) - x
    perm[..., -1] = good
    pr = perm / n
    obs = X / n[None, :]
    pbar = tot_pos / N
    t_gap_obs = gap(obs)
    t_gap = gap(pr)
    t_chi_obs = np.sum(n[None, :] * (obs - pbar[:, None]) ** 2, axis=1)
    t_chi = np.sum(n * (pr - pbar[:, None, None]) ** 2, axis=2)
    p_gap = (1 + np.sum(t_gap >= t_gap_obs[:, None] - TOL, axis=1)) / (B + 1)
    p_chi = (1 + np.sum(t_chi >= t_chi_obs[:, None] - TOL, axis=1)) / (B + 1)
    return p_gap, p_chi


def m4_m_out_of_n(rng, X, n, B, power: float = 0.7):
    """Stratified m-out-of-n bootstrap with a common resampling fraction.

    f = n_min**(power-1) so the smallest group draws n_min**power rows; every
    group uses m_k = max(2, round(f n_k)) so a single scale sqrt(m/n) applies.
    Interval from the root sqrt(m)(θ*_m − θ̂) ≈ sqrt(n)(θ̂ − θ), lower end clipped at 0.
    """
    n_min = n.min()
    f = n_min ** (power - 1.0)
    m = np.maximum(2, np.round(f * n)).astype(int)
    scale = np.sqrt(m[np.argmin(n)] / n_min)
    ghat = gap(X / n[None, :])
    g_star = gap(m1_rates(rng, X, n, B, m=m))
    root = g_star - ghat[:, None]
    q_lo, q_hi = np.quantile(root, [ALPHA / 2, 1 - ALPHA / 2], axis=1)
    lo = np.maximum(ghat - scale * q_hi, 0.0)
    hi = np.minimum(ghat - scale * q_lo, 1.0)
    return lo, hi
