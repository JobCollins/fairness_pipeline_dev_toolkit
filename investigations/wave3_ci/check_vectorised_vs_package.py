"""Cross-check the vectorised simulators in gapsim.py against the real package.

1. BCa: feed identical replicate vectors to gapsim.bca and stats.bootstrap.bca_ci;
   the intervals must agree to float precision (finite and NaN-containing cases).
2. M0: coverage / lower-bound distribution of FairnessAnalyzer (real index bootstrap)
   vs the count-level multinomial simulator, on two cells. Agreement is
   distributional, so it is judged against Monte Carlo error.

Run: PYTHONPATH=. .venv/bin/python investigations/wave3_ci/check_vectorised_vs_package.py
"""

from __future__ import annotations

import json
from pathlib import Path

import gapsim as G
import numpy as np

from fairness_pipeline_dev_toolkit.metrics.core import (
    FairnessAnalyzer,
    dpd_stat_from_indices,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import bca_ci

OUT = Path(__file__).resolve().parent / "results"


def bca_check() -> dict:
    out = {}
    for label, n, nan_inject in (
        ("finite", np.array([40, 60, 50]), 0),
        ("with NaN", np.array([40, 60, 50]), 7),
    ):
        rng = np.random.default_rng(5)
        X = rng.binomial(n, [0.3, 0.4, 0.35])[None, :]
        y = np.concatenate([np.r_[np.ones(X[0, k]), np.zeros(n[k] - X[0, k])] for k in range(3)])
        s = np.repeat(["0", "1", "2"], n)
        groups = ["0", "1", "2"]
        idx = np.arange(len(y))
        boot, _ = G.m0_replicates(np.random.default_rng(9), X, n, 1000)
        boot = boot.copy()
        boot[0, :nan_inject] = np.nan
        theta = np.array([dpd_stat_from_indices(idx, y, s, groups)])
        mine = G.bca(boot, theta, G.jackknife_accel(X, n), drop_nan=False)[:2]
        pkg = bca_ci(idx, lambda ii: dpd_stat_from_indices(ii, y, s, groups), boot[0], level=0.95)
        out[label] = {
            "gapsim": [float(mine[0][0]), float(mine[1][0])],
            "package": [float(pkg[0]), float(pkg[1])],
            "agree": bool(np.allclose([mine[0][0], mine[1][0]], pkg, equal_nan=True, atol=1e-12)),
        }
    return out


def m0_check(K, n_per, base, g, n_sim=300, B=1000) -> dict:
    n = np.full(K, n_per)
    p = np.full(K, base)
    p[0] = base + g
    fa = FairnessAnalyzer(min_group_size=30, backend="native")
    s = np.repeat([str(k) for k in range(K)], n_per)
    rng = np.random.default_rng(77)
    X = rng.binomial(n[None, :], p[None, :], size=(n_sim, K))
    pk_lo, pk_cov = [], 0
    for i in range(n_sim):
        y = np.concatenate([np.r_[np.ones(X[i, k]), np.zeros(n_per - X[i, k])] for k in range(K)])
        r = fa.demographic_parity_difference(y, s, ci_samples=B, with_effect_size=False)
        pk_lo.append(r.ci[0])
        pk_cov += r.ci[0] - 1e-12 <= g <= r.ci[1] + 1e-12
    g0, _ = G.m0_replicates(np.random.default_rng(78), X, n, B)
    lo, hi = G.percentile(g0)
    v_cov = np.mean((lo - 1e-12 <= g) & (g <= hi + 1e-12))
    se = np.sqrt(0.25 / n_sim)
    return {
        "cell": f"K={K} n={n_per} base={base} gap={g}",
        "package_coverage": pk_cov / n_sim,
        "vectorised_coverage": float(v_cov),
        "mc_se_bound": se,
        "package_median_lower": float(np.median(pk_lo)),
        "vectorised_median_lower": float(np.median(lo)),
    }


def main() -> None:
    res = {"bca": bca_check()}
    print(json.dumps(res["bca"], indent=1))
    res["m0"] = [m0_check(3, 100, 0.5, 0.0), m0_check(2, 100, 0.1, 0.1)]
    print(json.dumps(res["m0"], indent=1))
    (OUT / "check_vectorised_vs_package.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
