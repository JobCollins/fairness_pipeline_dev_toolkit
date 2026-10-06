"""Wave 3a PR B — rate-disparity CI candidates for refusal / toxicity / stereotype.

Design: template random effects, T ∈ {3,5,10,30}, K=3 groups, gap ∈ {0,.05,.1,.2}.
Binary outcomes (refusal / stereotype) and bounded continuous (toxicity in [0,1]).

Candidates:
  m2a     — Bonferroni Agresti–Caffo ignoring pairing (binary only)
  paired_t — per-template group-difference vectors, Bonferroni-t, C2b-style
             inversion via simultaneous_gap_bounds
  boot    — existing bootstrap_rate_disparity (within-group, B=2000)

Zero-variance rule: if every per-template group difference is exactly 0
(e.g. all-refused fixture), paired_t returns undefined:zero_variance rather
than a degenerate [0,0].

Run:  .venv/bin/python investigations/wave3a/sim_rate_disparity.py
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.llm_evals.scoring import bootstrap_rate_disparity
from fairness_pipeline_dev_toolkit.stats.gap_intervals import (
    binary_gap_interval,
    simultaneous_gap_bounds,
)

OUT = Path(__file__).resolve().parent / "results"
TS = (3, 5, 10, 30)
GAPS = (0.0, 0.05, 0.10, 0.20)
K = 3
S = 500
B = 2000
TOL = 1e-12


def _simulate_binary(rng, T, gap, base=0.3):
    # Template random effect on logit scale; group 0 elevated by gap on probability.
    te = rng.normal(0, 0.5, T)
    p = np.clip(base + te[:, None] * 0.1 + np.array([gap, 0, 0]), 0.01, 0.99)
    return rng.binomial(1, p)  # (T, K)


def _simulate_toxicity(rng, T, gap, base=0.2):
    te = rng.normal(0, 0.05, T)
    means = np.clip(base + te[:, None] + np.array([gap, 0, 0]), 0.01, 0.99)
    return np.clip(rng.normal(means, 0.05), 0, 1)


def _paired_t(mat, level=0.95):
    """mat shape (T, K); Bonferroni-t on per-template pairwise diffs, inverted."""
    T, k = mat.shape
    if T < 2:
        raise IntervalUndefinedError("too_few_templates", f"T={T} < 2")
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    a_tail = (1 - level) / (2 * len(pairs))
    from scipy.stats import t as student_t

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


def _m2a(mat, level=0.95):
    x = mat.sum(axis=0)
    n = np.full(mat.shape[1], mat.shape[0])
    return binary_gap_interval(x, n, level)


def _boot(mat, rng_seed, level=0.95):
    import warnings

    scores = {str(j): mat[:, j].tolist() for j in range(mat.shape[1])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return bootstrap_rate_disparity(scores, B=B, level=level, random_state=rng_seed)


def run_cell(args):
    kind, T, gap = args
    rng = np.random.default_rng([{"binary": 1, "toxicity": 2}[kind], T, int(gap * 100), 11])
    cov = {
        c: 0 for c in (("m2a", "paired_t", "boot") if kind == "binary" else ("paired_t", "boot"))
    }
    excl = {c: 0 for c in cov}
    undef = {c: 0 for c in cov}
    for s in range(S):
        mat = _simulate_binary(rng, T, gap) if kind == "binary" else _simulate_toxicity(rng, T, gap)
        point = float(mat.mean(axis=0).max() - mat.mean(axis=0).min())
        true = gap
        methods = {}
        if kind == "binary":
            try:
                methods["m2a"] = _m2a(mat)
            except IntervalUndefinedError:
                undef["m2a"] += 1
        try:
            methods["paired_t"] = _paired_t(mat)
        except IntervalUndefinedError:
            undef["paired_t"] += 1
        methods["boot"] = _boot(mat, int(rng.integers(2**31)))
        for name, bounds in methods.items():
            lo, hi = bounds
            if name != "boot":
                lo, hi = min(lo, point), max(hi, point)
            cov[name] += lo - TOL <= true <= hi + TOL
            excl[name] += not (lo - TOL <= point <= hi + TOL)
    out = {"kind": kind, "T": T, "gap": gap}
    for name in cov:
        out[f"{name}_cov"] = cov[name] / S
        out[f"{name}_excl"] = excl[name] / S
        out[f"{name}_undef"] = undef.get(name, 0) / S
    return out


def main():
    t0 = time.perf_counter()
    cells = [(k, T, g) for k in ("binary", "toxicity") for T in TS for g in GAPS]
    with Pool(8) as pool:
        rows = pool.map(run_cell, cells, chunksize=1)
    summary = {"runtime_s": time.perf_counter() - t0, "rows": rows}
    # Decision 8 at gap 0
    for kind in ("binary", "toxicity"):
        summary[kind] = {}
        for name in ("m2a", "paired_t", "boot"):
            gap0 = [r for r in rows if r["kind"] == kind and r["gap"] == 0.0 and f"{name}_cov" in r]
            if not gap0:
                continue
            mean_c = float(np.mean([r[f"{name}_cov"] for r in gap0]))
            min_c = float(np.min([r[f"{name}_cov"] for r in gap0]))
            excl = float(np.max([r[f"{name}_excl"] for r in gap0]))
            summary[kind][name] = {
                "mean": mean_c,
                "min": min_c,
                "excl_max": excl,
                "pass": bool(mean_c >= 0.95 and min_c >= 0.93 and excl == 0.0),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rate_disparity_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
