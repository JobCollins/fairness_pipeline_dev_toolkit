"""Part 3c — BCa policies for non-finite bootstrap replicates (BL-031).

Policies applied to the analyzer's unstratified (M0) replicates:
  P0  package as-is: any NaN replicate -> np.percentile returns NaN (CI = NaN);
      NaN jackknife (group of size 1) -> ValueError.
  P1  fall back to percentile (finite replicates) when any replicate or the
      acceleration is non-finite.
  P2  drop non-finite replicates, run BCa if the finite share >= floor
      (0.99 and 0.90 shown), else undefined; NaN acceleration -> undefined.
  P3  refuse: undefined whenever any replicate is non-finite.
Stratified alternatives: S-BCa (BCa on within-group replicates; NaN only via the
jackknife when a group has one row) and M2a for reference.

Part A: the exact Part 2 datasets (seed 31) and the exact package replicate vector.
Part B: simulation cells with a small minority group, S=500, B=1000.
Part C: EOD — does stratifying by group alone remove non-finite TPR/FPR? It does not,
        and eod_stat_from_indices silently drops the group instead of returning NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import gapsim as G
import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.metrics.core import (
    dpd_stat_from_indices,
    eod_stat_from_indices,
)

OUT = Path(__file__).resolve().parent / "results"
S, B = 500, 1000


def policies(boot, theta, accel):
    """boot (S,B), theta (S,), accel (S,) -> {policy: (lo, hi)}."""
    fin = np.isfinite(boot)
    share = fin.mean(axis=1)
    any_nan = share < 1
    acc_ok = np.isfinite(accel)
    p_lo, p_hi = G.percentile(boot)
    b_lo, b_hi, _ = G.bca(boot, theta, np.where(acc_ok, accel, 0.0), drop_nan=True)
    out = {}
    lo, hi, _ = G.bca(boot, theta, accel, drop_nan=False)
    out["P0 as-is"] = (lo, hi)
    fb = any_nan | ~acc_ok
    out["P1 fallback pct"] = (np.where(fb, p_lo, b_lo), np.where(fb, p_hi, b_hi))
    for floor in (0.99, 0.90):
        ok = (share >= floor) & acc_ok
        out[f"P2 drop>= {floor}"] = (np.where(ok, b_lo, np.nan), np.where(ok, b_hi, np.nan))
    ok = ~any_nan & acc_ok
    out["P3 refuse"] = (np.where(ok, b_lo, np.nan), np.where(ok, b_hi, np.nan))
    return out, share


def part_a():
    res = {}
    for m_minor in (1, 2, 3, 5):
        rng = np.random.default_rng(31)
        n_major = 300
        y = np.concatenate([rng.binomial(1, 0.5, n_major), rng.binomial(1, 0.5, m_minor)])
        s = np.array(["maj"] * n_major + ["min"] * m_minor)
        groups = ["maj", "min"]
        x = np.arange(len(y))
        bg = np.random.default_rng(42)
        boot = np.array(
            [
                dpd_stat_from_indices(x[bg.integers(0, len(x), len(x))], y, s, groups)
                for _ in range(B)
            ]
        )[None, :]
        n = np.array([n_major, m_minor])
        X = np.array([[y[:n_major].sum(), y[n_major:].sum()]])
        theta = G.gap(X / n)
        out, share = policies(boot, theta, G.jackknife_accel(X, n))
        strat = G.gap(G.m1_rates(np.random.default_rng(42), X, n, B))
        acc = G.jackknife_accel(X, n)
        s_lo, s_hi, _ = G.bca(strat, theta, acc, drop_nan=True)
        row = {
            "value": float(theta[0]),
            "nan_share": float(1 - share[0]),
            "accel_finite": bool(np.isfinite(acc[0])),
        }
        for k, (lo, hi) in out.items():
            row[k] = [round(float(lo[0]), 4), round(float(hi[0]), 4)]
        row["S-BCa (stratified)"] = [round(float(s_lo[0]), 4), round(float(s_hi[0]), 4)]
        res[f"minor={m_minor}"] = row
    return res


def part_b():
    rows = []
    for K, maj in ((2, [300]), (3, [300, 300])):
        for m in (1, 2, 3, 5, 10):
            n = np.array(maj + [m])
            for g in (0.0, 0.2):
                p = np.full(K, 0.5)
                p[0] += g  # elevated group is a majority group
                truth = p.max() - p.min()
                ss = np.random.SeedSequence([K, m, int(g * 10), 31])
                rd, r0, r1 = (np.random.default_rng(x) for x in ss.spawn(3))
                X = rd.binomial(n[None, :], p[None, :], size=(S, K))
                theta = G.gap(X / n[None, :])
                acc = G.jackknife_accel(X, n)
                boot, _ = G.m0_replicates(r0, X, n, B)
                out, share = policies(boot, theta, acc)
                strat = G.gap(G.m1_rates(r1, X, n, B))
                out["S-BCa (stratified)"] = G.bca(
                    strat, theta, np.where(np.isfinite(acc), acc, np.nan), drop_nan=True
                )[:2]
                out["M1 pct stratified"] = G.percentile(strat)
                out["M2a simult."] = G.m2_bonferroni_ac(X, n)
                for k, (lo, hi) in out.items():
                    d = np.isfinite(lo) & np.isfinite(hi)
                    cov = d & (lo <= truth + 1e-12) & (truth - 1e-12 <= hi)
                    rows.append(
                        {
                            "K": K,
                            "n_minor": m,
                            "gap": g,
                            "policy": k,
                            "mean_nan_share": float(1 - share.mean()),
                            "undefined": float(1 - d.mean()),
                            "coverage_uncond": float(cov.mean()),
                            "coverage_if_defined": float(cov.sum() / max(d.sum(), 1)),
                            "width": float(np.nanmean(np.where(d, hi - lo, np.nan))),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part3c_bca_nan_policy.csv", index=False)
    return df


def part_c():
    """EOD with a group that has few positives: silent group drop inside the bootstrap."""
    rng = np.random.default_rng(3)
    n_per = [200, 30]
    y_true = np.concatenate([rng.binomial(1, 0.5, 200), np.r_[1, 1, np.zeros(28)]])
    y_pred = rng.binomial(1, 0.5, len(y_true))
    s = np.repeat(["a", "b"], n_per)
    groups = ["a", "b"]
    idx_by_g = [np.where(s == g)[0] for g in groups]
    br = np.random.default_rng(4)
    dropped = nan = 0
    for _ in range(B):
        samp = np.concatenate([ix[br.integers(0, len(ix), len(ix))] for ix in idx_by_g])
        b_pos = np.sum((s[samp] == "b") & (y_true[samp] == 1))
        v = eod_stat_from_indices(samp, y_true, y_pred, s, groups)
        if not np.isfinite(v):
            nan += 1
        elif b_pos == 0:
            dropped += 1
    return {
        "design": "group b: 30 rows, 2 positives; group-stratified resample",
        "share_replicates_group_b_has_no_positives": dropped / B,
        "share_nan_returned": nan / B,
        "behaviour": "eod_stat_from_indices returns the FPR gap only (TPR gap silently "
        "undefined, nanmax picks FPR); no NaN, no flag. Stratifying by (group, y_true) "
        "keeps every cell non-empty and removes this.",
    }


def main():
    a = part_a()
    print(json.dumps(a, indent=1))
    df = part_b()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(
        df.pivot_table(
            index=["K", "n_minor", "gap"],
            columns="policy",
            values=["coverage_uncond", "undefined"],
        ).round(3)
    )
    c = part_c()
    print(json.dumps(c, indent=1))
    (OUT / "part3c_bca_nan_policy.json").write_text(json.dumps({"A": a, "C": c}, indent=1))


if __name__ == "__main__":
    main()
