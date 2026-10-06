"""Part 3a — coverage of max−min gap intervals across the full grid.

Grid: K in {2,3,5}; n per group in {30,100,1000} plus one unequal design;
base rate in {0.1,0.5}; true gap in {0,0.02,0.05,0.10,0.20}.
Truth: group 0 has rate base+gap, all other groups have rate base (one elevated
group). In the unequal design the elevated group is the *smallest* one.
Unequal designs: K=2 (30, 3000), K=3 (30, 300, 3000), K=5 (30, 100, 300, 1000, 3000).

S=500 simulated datasets per cell, B=1000 replicates for every bootstrap/permutation.
Common random numbers: every method sees the same S datasets in a cell.

Run: PYTHONPATH=. .venv/bin/python investigations/wave3_ci/part3a_gap_grid.py
"""

from __future__ import annotations

import time
from pathlib import Path

import gapsim as G
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

S = 500
B = 1000
KS = (2, 3, 5)
NS = ("30", "100", "1000", "unequal")
BASES = (0.1, 0.5)
GAPS = (0.0, 0.02, 0.05, 0.10, 0.20)
UNEQUAL = {2: [30, 3000], 3: [30, 300, 3000], 5: [30, 100, 300, 1000, 3000]}


def sizes(K: int, ns: str) -> np.ndarray:
    return np.array(UNEQUAL[K] if ns == "unequal" else [int(ns)] * K)


def run_cell(K: int, ns: str, base: float, g: float) -> list[dict]:
    n = sizes(K, ns)
    p = np.full(K, base)
    p[0] = base + g
    true_gap = p.max() - p.min()
    seed = np.random.SeedSequence([K, NS.index(ns), BASES.index(base), GAPS.index(g), 314])
    r_data, r0, r1, r3, r4 = (np.random.default_rng(s) for s in seed.spawn(5))
    X = r_data.binomial(n[None, :], p[None, :], size=(S, K))
    ghat = G.gap(X / n[None, :])

    rows: list[dict] = []

    def record(method, lo, hi, secs, extra=None):
        defined = np.isfinite(lo) & np.isfinite(hi)
        cov = (lo <= true_gap + G.TOL) & (true_gap - G.TOL <= hi) & defined
        excl0 = (lo > G.TOL) & defined
        excl_pt = ((ghat < lo - G.TOL) | (ghat > hi + G.TOL)) & defined
        row = {
            "K": K,
            "n": ns,
            "base": base,
            "gap": g,
            "method": method,
            "coverage": cov.mean(),
            "coverage_mcse": np.sqrt(cov.mean() * (1 - cov.mean()) / S),
            "excl0": excl0.mean(),
            "excl0_mcse": np.sqrt(excl0.mean() * (1 - excl0.mean()) / S),
            "width": np.nanmean(np.where(defined, hi - lo, np.nan)),
            "undefined": 1 - defined.mean(),
            "excl_point": excl_pt.mean(),
            "ms_per_dataset": 1000 * secs / S,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    t = time.perf_counter()
    g0, tot0 = G.m0_replicates(r0, X, n, B)
    lo, hi = G.percentile(g0)
    record("M0 pct unstratified", lo, hi, time.perf_counter() - t)
    t0_rep = time.perf_counter() - t

    t = time.perf_counter()
    acc = G.jackknife_accel(X, n)
    t_acc = time.perf_counter() - t
    t = time.perf_counter()
    lo, hi, info = G.bca(g0, ghat, acc, drop_nan=False)
    record("M5a BCa as-is (M0 reps)", lo, hi, t0_rep + t_acc + time.perf_counter() - t)
    t = time.perf_counter()
    lo, hi, info = G.bca(g0, ghat, acc, drop_nan=True)
    record(
        "M5b BCa drop-NaN (M0 reps)",
        lo,
        hi,
        t0_rep + t_acc + time.perf_counter() - t,
        {"nan_rep_share": float(np.mean(info["nan_share"]))},
    )

    t = time.perf_counter()
    r1s = G.m1_rates(r1, X, n, B)
    g1 = G.gap(r1s)
    t1_rep = time.perf_counter() - t
    t = time.perf_counter()
    lo, hi = G.percentile(g1)
    record("M1 pct stratified", lo, hi, t1_rep + time.perf_counter() - t)
    t = time.perf_counter()
    lo, hi, _ = G.bca(g1, ghat, acc, drop_nan=True)
    record("M5c BCa (M1 reps)", lo, hi, t1_rep + t_acc + time.perf_counter() - t)

    t = time.perf_counter()
    lo, hi = G.m2_bonferroni_ac(X, n)
    record("M2a simult. Bonferroni-AC", lo, hi, time.perf_counter() - t)
    t = time.perf_counter()
    lo, hi = G.m2_maxt(X, n, r1s)
    record("M2b simult. boot max-|t|", lo, hi, t1_rep + time.perf_counter() - t)

    t = time.perf_counter()
    lo, hi = G.m4_m_out_of_n(r4, X, n, B)
    record("M4 m-out-of-n (n^0.7)", lo, hi, time.perf_counter() - t)

    t = time.perf_counter()
    p_gap, p_chi = G.m3_permutation(r3, X, n, B)
    secs = time.perf_counter() - t
    for name, pv in (("M3 perm test (gap stat)", p_gap), ("M3 perm test (chi2 stat)", p_chi)):
        rej = (pv <= G.ALPHA).mean()
        rows.append(
            {
                "K": K,
                "n": ns,
                "base": base,
                "gap": g,
                "method": name,
                "reject": rej,
                "reject_mcse": np.sqrt(rej * (1 - rej) / S),
                "ms_per_dataset": 1000 * secs / S,
            }
        )
    return rows


def main() -> None:
    all_rows = []
    t_start = time.perf_counter()
    for K in KS:
        for ns in NS:
            for base in BASES:
                for g in GAPS:
                    all_rows.extend(run_cell(K, ns, base, g))
            print(f"K={K} n={ns} done ({time.perf_counter() - t_start:.0f}s)", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "part3a_gap_grid.csv", index=False)
    print(f"wrote {len(df)} rows in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
