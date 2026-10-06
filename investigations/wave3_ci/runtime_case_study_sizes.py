"""Runtime: today's analyzer CI vs prototypes of the candidate methods on case-study sizes.

Datasets
- COMPAS: case_studies/compas_bw.csv (5,278 rows, 2 race groups) — real file.
- ACS-like: synthetic 39,321 rows (20% test split of 196,604), 5 groups with shares
  .60/.15/.15/.06/.04 — a size proxy only, the ACS parquet is not committed.

Prototypes operate on row-level arrays (not the binary-count shortcut used in the
simulations) so the timings reflect what a package implementation would cost for
DPD, and by extension EOD/MAE (same group-mean kernel with a different value column).

LLM: hiring fixture shape (T=9 templates, 3 pairs each) — current vs C1 / C2b.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as tdist

from fairness_pipeline_dev_toolkit.metrics.core import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"


def timeit(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def stratified_pct(y, codes, K, B, rng):
    """M1 prototype: within-group index resampling, group means via bincount."""
    idx_by = [np.flatnonzero(codes == k) for k in range(K)]
    out = np.empty(B)
    for b in range(B):
        means = [y[ix[rng.integers(0, ix.size, ix.size)]].mean() for ix in idx_by]
        out[b] = max(means) - min(means)
    return np.percentile(out, [2.5, 97.5])


def m2a_ac(y, codes, K):
    n = np.bincount(codes, minlength=K)
    x = np.bincount(codes, weights=y, minlength=K)
    pt = (x + 1) / (n + 2)
    pairs = list(combinations(range(K), 2))
    z = norm.ppf(1 - 0.05 / (2 * len(pairs)))
    lb = ub = 0.0
    for i, j in pairs:
        d = pt[i] - pt[j]
        se = np.sqrt(pt[i] * (1 - pt[i]) / (n[i] + 2) + pt[j] * (1 - pt[j]) / (n[j] + 2))
        L, U = d - z * se, d + z * se
        lb = max(lb, 0.0, L, -U)
        ub = max(ub, abs(L), abs(U))
    return lb, min(ub, 1.0)


def permutation_gap(y, codes, K, B, rng):
    """M3 prototype: permute group labels, recompute max−min of group means."""
    n = np.bincount(codes, minlength=K)
    obs = np.bincount(codes, weights=y, minlength=K) / n
    t_obs = obs.max() - obs.min()
    hits = 0
    for _ in range(B):
        p = rng.permutation(codes)
        m = np.bincount(p, weights=y, minlength=K) / n
        hits += (m.max() - m.min()) >= t_obs - 1e-12
    return (1 + hits) / (B + 1)


def run_dataset(name, y, s):
    codes, uniq = pd.factorize(s)
    K = len(uniq)
    fa = FairnessAnalyzer(min_group_size=30, backend="native")
    rng = np.random.default_rng(0)
    res = {"dataset": name, "rows": int(len(y)), "groups": int(K)}
    res["analyzer DPD no CI (s)"] = timeit(
        lambda: fa.demographic_parity_difference(y, s, with_ci=False, with_effect_size=False)
    )
    res["analyzer DPD percentile B=1000 (s) [today]"] = timeit(
        lambda: fa.demographic_parity_difference(y, s, ci_samples=1000, with_effect_size=False), 1
    )
    res["M1 stratified pct B=1000 (s)"] = timeit(lambda: stratified_pct(y, codes, K, 1000, rng))
    res["M2a Bonferroni-AC (s)"] = timeit(lambda: m2a_ac(y, codes, K))
    res["M3 permutation B=1000 (s)"] = timeit(lambda: permutation_gap(y, codes, K, 1000, rng))
    res["M3 permutation B=2000 (s)"] = timeit(lambda: permutation_gap(y, codes, K, 2000, rng), 1)
    if len(y) <= 6000:
        t = time.perf_counter()
        fa.demographic_parity_difference(
            y, s, ci_method="bca", ci_samples=1000, with_effect_size=False
        )
        res["analyzer DPD BCa B=1000 (s) [today, O(N^2) jackknife]"] = time.perf_counter() - t
    return res


def llm_timing():
    rng = np.random.default_rng(1)
    T, D = 9, 1
    pairs = rng.normal(0.2, 0.03, size=(T, D + 1, 3))
    tm = pairs.mean(axis=2)
    pooled = pairs[:, :D, :].reshape(-1)
    res = {"shape": "T=9, D=1 gated + control, 3 pairs per bucket"}
    res["C0 today: bootstrap_ci(np.mean) B=200 (s)"] = timeit(
        lambda: bootstrap_ci(pooled, np.mean, B=200, random_state=42)
    )

    def c1(B):
        W = rng.multinomial(T, np.full(T, 1 / T), size=B) / T
        star = W @ tm[:, :D]
        return np.percentile(star.max(axis=1), [2.5, 97.5])

    def c2b():
        x = tm[:, :D] - tm[:, D:]
        m = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(T)
        c = tdist.ppf(1 - 0.05 / (2 * D), T - 1)
        return (m - c * se).max(), (m + c * se).max()

    res["C1 cluster B=2000 (s)"] = timeit(lambda: c1(2000))
    res["C2b Bonferroni-t (s)"] = timeit(c2b)
    return res


def main():
    compas = pd.read_csv(ROOT / "case_studies" / "compas_bw.csv")
    rows = [run_dataset("COMPAS (real)", compas["y_pred"].to_numpy(), compas["race"].to_numpy())]
    rng = np.random.default_rng(2)
    N = 39_321
    s = rng.choice(["W", "H", "A", "B", "O"], size=N, p=[0.60, 0.15, 0.15, 0.06, 0.04])
    y = rng.binomial(1, 0.3, N)
    rows.append(run_dataset("ACS-like (synthetic size proxy)", y, s))
    out = {"classical": rows, "llm": llm_timing()}
    (OUT / "runtime_case_study_sizes.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
