"""Wave 3a PR A — calibration of the classifier gap intervals and permutation p-values.

Calls the *package* functions (``stats.gap_intervals``) so the numbers validate the
shipped code, not a re-implementation.

Grid (the Wave 3a investigation gap grid):
  K in {2, 3, 5}; n per group in {30, 100, 1000} plus an unequal design in which the
  elevated group is the smallest one; true gap in {0, .02, .05, .10, .20}.
  Unequal sizes: K=2 (30, 3000), K=3 (30, 300, 3000), K=5 (30, 100, 300, 1000, 3000).

Metrics and candidates:
  DPD  — M2a ``binary_gap_interval``; base rate in {0.1, 0.5}; group 0 rate base+gap.
  EOD  — ``equalized_odds_gap_interval`` (Agresti–Caffo over TPR and FPR pairs, one
         Bonferroni over both families); prevalence 0.4, FPR = base, TPR = 1 − base,
         base in {0.1, 0.5}; group 0 TPR lowered by gap (true EOD = gap).
  MAE  — ``welch_gap_interval`` (Bonferroni Welch-t on per-row absolute errors);
         absolute errors half-normal, exponential or lognormal(σ=1) with mean 1,
         group 0 mean 1 + gap.

Coverage: S=500 datasets per cell. Permutation type-I error at gap 0: S=2000 per
cell, ``permutation_gap_pvalue`` with its default 2000 shuffles. Power: S=500.

Run (repo root):  .venv/bin/python investigations/wave3a/sim_classifier_gaps.py
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.stats.gap_intervals import (
    binary_gap_interval,
    equalized_odds_gap_interval,
    permutation_gap_pvalue,
    welch_gap_interval,
)

OUT = Path(__file__).resolve().parent / "results"
S_COV = 500
S_TYPE1 = 2000
S_POWER = 500
KS = (2, 3, 5)
NS = ("30", "100", "1000", "unequal")
GAPS = (0.0, 0.02, 0.05, 0.10, 0.20)
UNEQUAL = {2: [30, 3000], 3: [30, 300, 3000], 5: [30, 100, 300, 1000, 3000]}
SETTINGS = {"dpd": (0.1, 0.5), "eod": (0.1, 0.5), "mae": ("halfnormal", "exponential", "lognormal")}
PREVALENCE = 0.4
TOL = 1e-12


def sizes(k: int, ns: str) -> np.ndarray:
    return np.array(UNEQUAL[k] if ns == "unequal" else [int(ns)] * k)


def _abs_errors(rng, dist: str, mean: float, size) -> np.ndarray:
    if dist == "halfnormal":
        return np.abs(rng.normal(0.0, mean / np.sqrt(2 / np.pi), size))
    if dist == "exponential":
        return rng.exponential(mean, size)
    return rng.lognormal(np.log(mean) - 0.5, 1.0, size)


def simulate(metric: str, setting, k: int, ns: str, gap: float, s: int, rng):
    """Yield (point_estimate, interval_fn, rows_fn) per simulated dataset."""
    n = sizes(k, ns)
    out = []
    if metric == "dpd":
        p = np.full(k, setting)
        p[0] += gap
        x = rng.binomial(n[None, :], p[None, :], size=(s, k))
        for row in x:
            rates = row / n
            out.append(
                (
                    rates.max() - rates.min(),
                    lambda row=row: binary_gap_interval(row, n),
                    lambda row=row: _binary_rows(row, n),
                )
            )
    elif metric == "eod":
        fpr = np.full(k, setting)
        tpr = np.full(k, 1 - setting)
        tpr[0] -= gap
        pos = rng.binomial(n[None, :], PREVALENCE, size=(s, k))
        neg = n[None, :] - pos
        tp = rng.binomial(pos, tpr[None, :])
        fp = rng.binomial(neg, fpr[None, :])
        for a, b, c, d in zip(tp, pos, fp, neg):
            with np.errstate(invalid="ignore", divide="ignore"):
                t_r, f_r = a / b, c / d
            est = max(np.nanmax(t_r) - np.nanmin(t_r), np.nanmax(f_r) - np.nanmin(f_r))
            out.append(
                (
                    est,
                    lambda a=a, b=b, c=c, d=d: equalized_odds_gap_interval(a, b, c, d),
                    lambda a=a, b=b, c=c, d=d: _eod_rows(a, b, c, d),
                )
            )
    else:
        means = np.ones(k)
        means[0] += gap
        draws = [_abs_errors(rng, setting, means[j], (s, n[j])) for j in range(k)]
        for i in range(s):
            groups = [d[i] for d in draws]
            m = np.array([g.mean() for g in groups])
            v = np.array([g.var(ddof=1) for g in groups])
            out.append(
                (
                    m.max() - m.min(),
                    lambda m=m, v=v: welch_gap_interval(m, v, n),
                    lambda groups=groups: _cont_rows(groups),
                )
            )
    return out


def _binary_rows(x, n):
    vals = np.concatenate([np.r_[np.ones(xi), np.zeros(ni - xi)] for xi, ni in zip(x, n)])
    codes = np.repeat(np.arange(len(n)), n)
    return vals, codes, None


def _eod_rows(tp, pos, fp, neg):
    vals, codes, strata = [], [], []
    for j, (a, b, c, d) in enumerate(zip(tp, pos, fp, neg)):
        vals += [np.ones(a), np.zeros(b - a), np.ones(c), np.zeros(d - c)]
        codes.append(np.full(b + d, j))
        strata += [np.ones(b), np.zeros(d)]
    return np.concatenate(vals), np.concatenate(codes), np.concatenate(strata).astype(int)


def _cont_rows(groups):
    return (
        np.concatenate(groups),
        np.concatenate([np.full(g.size, j) for j, g in enumerate(groups)]),
        None,
    )


def true_gap(metric: str, gap: float) -> float:
    return gap


def run_cell(args):
    metric, setting, k, ns, gap = args
    si = SETTINGS[metric].index(setting)
    seed = np.random.SeedSequence(
        [KS.index(k), NS.index(ns), si, GAPS.index(gap), 2026, len(metric)]
    )
    r_cov, r_t1, r_pow = (np.random.default_rng(x) for x in seed.spawn(3))
    tg = true_gap(metric, gap)
    row = {"metric": metric, "setting": setting, "K": k, "n": ns, "gap": gap}

    cov = undefined = excl_pt = 0
    widths = []
    t0 = time.perf_counter()
    for est, interval, _ in simulate(metric, setting, k, ns, gap, S_COV, r_cov):
        try:
            lo, hi = interval()
        except IntervalUndefinedError:
            undefined += 1
            continue
        cov += lo - TOL <= tg <= hi + TOL
        excl_pt += not (lo - TOL <= est <= hi + TOL)
        widths.append(hi - lo)
    cv = cov / S_COV
    row.update(
        coverage=cv,
        coverage_mcse=np.sqrt(cv * (1 - cv) / S_COV),
        excl_point=excl_pt / S_COV,
        undefined=undefined / S_COV,
        width=float(np.mean(widths)) if widths else np.nan,
        ms_interval=1000 * (time.perf_counter() - t0) / S_COV,
    )

    s_p = S_TYPE1 if gap == 0.0 else (S_POWER if metric != "mae" or gap == 0.10 else 0)
    if s_p:
        rng = r_t1 if gap == 0.0 else r_pow
        rej = 0
        t0 = time.perf_counter()
        for _, _, rows in simulate(metric, setting, k, ns, gap, s_p, rng):
            vals, codes, strata = rows()
            p = permutation_gap_pvalue(
                vals, codes, strata=strata, random_state=int(rng.integers(2**31))
            )
            rej += p <= 0.05
        rr = rej / s_p
        row.update(
            reject_rate=rr,
            reject_mcse=np.sqrt(rr * (1 - rr) / s_p),
            perm_sims=s_p,
            ms_pvalue=1000 * (time.perf_counter() - t0) / s_p,
        )
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [
        (m, st, k, ns, g)
        for m, sts in SETTINGS.items()
        for st in sts
        for k in KS
        for ns in NS
        for g in GAPS
    ]
    # Big MAE permutation cells first so the pool stays busy.
    cells.sort(key=lambda c: (c[0] != "mae" or c[4] not in (0.0, 0.10), c[3] != "1000"))
    t0 = time.perf_counter()
    with Pool(8) as pool:
        rows = pool.map(run_cell, cells, chunksize=1)
    df = pd.DataFrame(rows).sort_values(
        ["metric", "setting", "K", "n", "gap"], key=lambda c: c.astype(str)
    )
    df.to_csv(OUT / "classifier_gaps.csv", index=False)
    summary = {}
    for metric, d in df.groupby("metric"):
        z = d[d.gap == 0.0]
        summary[metric] = {
            "gap0_coverage_mean": float(z.coverage.mean()),
            "gap0_coverage_min": float(z.coverage.min()),
            "coverage_min_all_gaps": float(d.coverage.min()),
            "excl_point_max": float(d.excl_point.max()),
            "undefined_max": float(d.undefined.max()),
            "type1_max": float(z.reject_rate.max()),
            "type1_mean": float(z.reject_rate.mean()),
            "type1_mcse": float(np.sqrt(0.05 * 0.95 / S_TYPE1)),
            "pass_decision8": bool(
                z.coverage.mean() >= 0.95
                and z.coverage.min() >= 0.93
                and d.excl_point.max() == 0
                and z.reject_rate.max() <= 0.06
            ),
        }
    summary["runtime_s"] = time.perf_counter() - t0
    (OUT / "classifier_gaps_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
