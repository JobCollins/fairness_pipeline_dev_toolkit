"""Wave 3a PR B follow-up — toxicity paired-t at S=4000 (decision 8).

Re-runs only the gap-0 toxicity cells with the paired Bonferroni-t candidate
at S=4000 (the S=500 mean was 0.9495, within one MC SE of 0.95).

Run:  .venv/bin/python investigations/wave3a/sim_toxicity_paired_s4000.py
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.stats.gap_intervals import simultaneous_gap_bounds

OUT = Path(__file__).resolve().parent / "results"
TS = (3, 5, 10, 30)
S = 4000
TOL = 1e-12
K = 3


def _simulate_toxicity(rng, T, gap=0.0, base=0.2):
    te = rng.normal(0, 0.05, T)
    means = np.clip(base + te[:, None] + np.array([gap, 0, 0]), 0.01, 0.99)
    return np.clip(rng.normal(means, 0.05), 0, 1)


def _paired_t(mat, level=0.95):
    T, k = mat.shape
    if T < 2:
        raise IntervalUndefinedError("too_few_templates", f"T={T} < 2")
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    a_tail = (1 - level) / (2 * len(pairs))
    crit = float(student_t.ppf(1 - a_tail, T - 1))
    lo, hi = [], []
    for i, j in pairs:
        d = mat[:, i] - mat[:, j]
        sd = float(d.std(ddof=1))
        if sd <= 0:
            raise IntervalUndefinedError("zero_variance", "all template diffs equal")
        se = sd / np.sqrt(T)
        m = float(d.mean())
        lo.append(m - crit * se)
        hi.append(m + crit * se)
    return simultaneous_gap_bounds(lo, hi, max_gap=1.0)


def run_cell(T: int):
    rng = np.random.default_rng([2, T, 0, 11])
    cov = excl = undef = 0
    for _ in range(S):
        mat = _simulate_toxicity(rng, T, gap=0.0)
        point = float(mat.mean(axis=0).max() - mat.mean(axis=0).min())
        try:
            lo, hi = _paired_t(mat)
            lo, hi = min(lo, point), max(hi, point)
        except IntervalUndefinedError:
            undef += 1
            continue
        cov += lo - TOL <= 0.0 <= hi + TOL
        excl += not (lo - TOL <= point <= hi + TOL)
    cv = cov / S
    return {
        "T": T,
        "S": S,
        "coverage": cv,
        "excl_point": excl / S,
        "undefined": undef / S,
        "mcse": float(np.sqrt(cv * (1 - cv) / S)),
    }


def main():
    t0 = time.perf_counter()
    with Pool(4) as pool:
        rows = pool.map(run_cell, TS)
    mean_c = float(np.mean([r["coverage"] for r in rows]))
    min_c = float(np.min([r["coverage"] for r in rows]))
    excl = float(np.max([r["excl_point"] for r in rows]))
    summary = {
        "mean": mean_c,
        "min": min_c,
        "excl_max": excl,
        "pass_decision8": bool(mean_c >= 0.95 and min_c >= 0.93 and excl == 0.0),
        "runtime_s": time.perf_counter() - t0,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toxicity_paired_s4000.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
