"""Wave 3a PR A — follow-up to ``sim_classifier_gaps.py``.

1. EOD gap-0 coverage at S=4000 (the S=500 minimum, 0.928, was within one MC SE of
   the 0.93 floor).
2. MAE gap candidates at gap 0 on every design (S=500 per cell):
   - ``welch``: the Bonferroni Welch-t interval (``welch_gap_interval``);
   - ``boot_t``: pairwise studentized bootstrap (within-group resampling, B=2000,
     Bonferroni over pairs, inverted with ``simultaneous_gap_bounds``);
   - ``edgeworth``: Welch with a one-term Edgeworth (Johnson/Hall) skewness
     correction of the studentized-difference quantiles, Bonferroni, inverted.

Run (repo root):  .venv/bin/python investigations/wave3a/sim_followup_eod_mae.py
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from fairness_pipeline_dev_toolkit.stats.gap_intervals import (
    equalized_odds_gap_interval,
    simultaneous_gap_bounds,
    welch_gap_interval,
)

OUT = Path(__file__).resolve().parent / "results"
KS = (2, 3, 5)
NS = ("30", "100", "1000", "unequal")
UNEQUAL = {2: [30, 3000], 3: [30, 300, 3000], 5: [30, 100, 300, 1000, 3000]}
DISTS = ("halfnormal", "exponential", "lognormal")
B = 2000
TOL = 1e-12


def sizes(k, ns):
    return np.array(UNEQUAL[k] if ns == "unequal" else [int(ns)] * k)


def abs_errors(rng, dist, mean, size):
    if dist == "halfnormal":
        return np.abs(rng.normal(0.0, mean / np.sqrt(2 / np.pi), size))
    if dist == "exponential":
        return rng.exponential(mean, size)
    return rng.lognormal(np.log(mean) - 0.5, 1.0, size)


def eod_cell(args):
    base, k, ns, s = args
    rng = np.random.default_rng([KS.index(k), NS.index(ns), int(base * 10), 77])
    n = sizes(k, ns)
    pos = rng.binomial(n[None, :], 0.4, size=(s, k))
    neg = n[None, :] - pos
    tp = rng.binomial(pos, 1 - base)
    fp = rng.binomial(neg, base)
    cov = sum(
        equalized_odds_gap_interval(a, b, c, d)[0] <= TOL for a, b, c, d in zip(tp, pos, fp, neg)
    )
    cv = cov / s
    return {
        "base": base,
        "K": k,
        "n": ns,
        "S": s,
        "coverage": cv,
        "mcse": np.sqrt(cv * (1 - cv) / s),
    }


def boot_t_interval(groups, rng, level=0.95):
    k = len(groups)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    alpha = (1 - level) / len(pairs)
    m = np.array([g.mean() for g in groups])
    se2 = np.array([g.var(ddof=1) / g.size for g in groups])
    bm, bse2 = [], []
    for g in groups:
        idx = rng.integers(0, g.size, (B, g.size))
        r = g[idx]
        bm.append(r.mean(axis=1))
        bse2.append(r.var(axis=1, ddof=1) / g.size)
    lo, hi = np.empty(len(pairs)), np.empty(len(pairs))
    for p, (i, j) in enumerate(pairs):
        d = m[i] - m[j]
        se = np.sqrt(se2[i] + se2[j])
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (bm[i] - bm[j] - d) / np.sqrt(bse2[i] + bse2[j])
        t = t[np.isfinite(t)]
        q_lo, q_hi = np.quantile(t, [alpha / 2, 1 - alpha / 2])
        lo[p], hi[p] = d - q_hi * se, d - q_lo * se
    return simultaneous_gap_bounds(lo, hi, max_gap=None)


def edgeworth_interval(groups, level=0.95):
    k = len(groups)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    a = (1 - level) / len(pairs)
    z = norm.ppf([a / 2, 1 - a / 2])
    m = np.array([g.mean() for g in groups])
    n = np.array([g.size for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    mu3 = np.array([((g - g.mean()) ** 3).mean() for g in groups])
    lo, hi = [], []
    for i, j in pairs:
        se = np.sqrt(v[i] / n[i] + v[j] / n[j])
        lam = (mu3[i] / n[i] ** 2 - mu3[j] / n[j] ** 2) / se**3
        t = z - lam / 6 * (2 * z**2 + 1)
        d = m[i] - m[j]
        lo.append(d - se * t[1])
        hi.append(d - se * t[0])
    return simultaneous_gap_bounds(np.array(lo), np.array(hi), max_gap=None)


def mae_cell(args):
    dist, k, ns, s = args
    rng = np.random.default_rng([KS.index(k), NS.index(ns), DISTS.index(dist), 99])
    n = sizes(k, ns)
    out = {"dist": dist, "K": k, "n": ns, "S": s}
    cov = {"welch": 0, "boot_t": 0, "edgeworth": 0}
    t_boot = 0.0
    for _ in range(s):
        groups = [abs_errors(rng, dist, 1.0, n[j]) for j in range(k)]
        m = np.array([g.mean() for g in groups])
        v = np.array([g.var(ddof=1) for g in groups])
        cov["welch"] += welch_gap_interval(m, v, n)[0] <= TOL
        cov["edgeworth"] += edgeworth_interval(groups)[0] <= TOL
        t0 = time.perf_counter()
        cov["boot_t"] += boot_t_interval(groups, rng)[0] <= TOL
        t_boot += time.perf_counter() - t0
    for name, c in cov.items():
        out[name] = c / s
    out["ms_boot_t"] = 1000 * t_boot / s
    return out


def main():
    t0 = time.perf_counter()
    with Pool(8) as pool:
        mae = pool.map(
            mae_cell, [(d, k, ns, 500) for d in DISTS for k in KS for ns in NS], chunksize=1
        )
        eod = pool.map(eod_cell, [(b, k, ns, 4000) for b in (0.1, 0.5) for k in KS for ns in NS])
    mae, eod = pd.DataFrame(mae), pd.DataFrame(eod)
    mae.to_csv(OUT / "followup_mae_candidates.csv", index=False)
    eod.to_csv(OUT / "followup_eod_s4000.csv", index=False)
    summary = {
        "eod_s4000": {"mean": eod.coverage.mean(), "min": eod.coverage.min()},
        "mae": {
            c: {"mean": mae[c].mean(), "min": mae[c].min()}
            for c in ("welch", "boot_t", "edgeworth")
        },
        "runtime_s": time.perf_counter() - t0,
    }
    (OUT / "followup_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(mae.sort_values("boot_t").head(10).to_string())
    print(eod.sort_values("coverage").head(5).to_string())


if __name__ == "__main__":
    main()
