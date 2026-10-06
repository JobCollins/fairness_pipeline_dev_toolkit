"""Part 3b — LLM divergence / contrast CIs under template dependence.

Generator (matches the evaluator's data shape): T templates; D gated dimensions plus
one control arm; 3 groups per arm -> 3 matched pairs per (template, arm), and each
response is reused by 2 of the 3 pairs in its bucket. Pair value

    d[t,a,(g,h)] = mu_a + sigma * ( sqrt(rho) u_t
                   + sqrt(1-rho) * ( sqrt(.5) (s[t,a,g] + s[t,a,h]) / sqrt(2) + sqrt(.5) e ) )

u_t is a template effect shared by *all* arms (gated and control); s is a
response effect shared by the two pairs that reuse that response. Unit variance
inside the bracket, so rho is the within-template intraclass correlation.
sigma=0.03, mu=0.20 (scale of the hiring/humanitarian fixtures).

Estimands: divergence = max_a mu_a over gated arms; contrast = max_a mu_a - mu_control.
Means: all equal (truth: divergence 0.20, contrast 0), or gated arm 0 higher by delta.

Methods:
  C0   current code: iid percentile bootstrap of the pooled gated-pair mean
       (divergence) / independent difference of pooled means (contrast).
  C1   template-cluster bootstrap, recomputing the exact reported statistic, percentile.
  C2a  C1 + simultaneous inversion: cluster-bootstrap max-|t| intervals for every
       gated mu_a (or mu_a - mu_c, paired by template), then [max L_a, max U_a].
  C2b  same inversion with analytic Bonferroni-t on template-level means (df=T-1).
  C2c  C1 with an m-out-of-n template bootstrap (m = T^0.7), root interval.

S=500 datasets per cell, B=1000. Run:
  PYTHONPATH=.:investigations/wave3_ci .venv/bin/python investigations/wave3_ci/part3b_llm_template_sim.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

S, B = 500, 1000
ALPHA = 0.05
SIGMA, MU = 0.03, 0.20
TS = (5, 10, 20, 50)
RHOS = (0.0, 0.3, 0.7)
DS = (1, 2, 3)
DELTAS = (0.0, 0.01, 0.03)
PAIRS = ((0, 1), (0, 2), (1, 2))
CHUNK = 50


def generate(rng, T, D, rho, delta):
    """Pair values, shape (S, T, A, 3) with A = D gated arms + 1 control (last)."""
    A = D + 1
    mu = np.full(A, MU)
    mu[0] += delta
    u = rng.standard_normal((S, T, 1, 1))
    s = rng.standard_normal((S, T, A, 3))
    e = rng.standard_normal((S, T, A, 3))
    shared = np.stack([(s[..., g] + s[..., h]) / np.sqrt(2) for g, h in PAIRS], axis=-1)
    noise = np.sqrt(rho) * u + np.sqrt(1 - rho) * (np.sqrt(0.5) * shared + np.sqrt(0.5) * e)
    return mu[None, None, :, None] + SIGMA * noise, mu


def stats_from_arm_means(m, D):
    """m (..., A) arm means -> (divergence, contrast)."""
    gmax = m[..., :D].max(axis=-1)
    return gmax, gmax - m[..., D]


def pct(x):
    return np.percentile(x, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=1)


def run_cell(T, D, rho, delta):
    seed = np.random.SeedSequence([T, D, int(rho * 10), int(delta * 1000), 2718])
    r_data, r0, r1, r2 = (np.random.default_rng(x) for x in seed.spawn(4))
    d, mu = generate(r_data, T, D, rho, delta)
    truth_div = mu[:D].max()
    truth_con = truth_div - mu[D]
    tm = d.mean(axis=3)  # template-level arm means (S, T, A)
    arm = tm.mean(axis=1)  # (S, A)
    v_div, v_con = stats_from_arm_means(arm, D)

    res = {}
    timings = {}

    # ---- C0 (current code), chunked iid resampling of pooled pairs
    t0 = time.perf_counter()
    gated = d[:, :, :D, :].reshape(S, -1)
    ctrl = d[:, :, D, :].reshape(S, -1)
    c0_div = np.empty((S, 2))
    c0_con = np.empty((S, 2))
    for a in range(0, S, CHUNK):
        sl = slice(a, a + CHUNK)
        ns = gated[sl].shape[0]
        ig = r0.integers(0, gated.shape[1], size=(ns, B, gated.shape[1]))
        ic = r0.integers(0, ctrl.shape[1], size=(ns, B, ctrl.shape[1]))
        mg = np.take_along_axis(gated[sl][:, None, :], ig, axis=2).mean(axis=2)
        mc = np.take_along_axis(ctrl[sl][:, None, :], ic, axis=2).mean(axis=2)
        c0_div[sl] = pct(mg).T
        c0_con[sl] = pct(mg - mc).T
    res["C0 current"] = (c0_div, c0_con)
    timings["C0 current"] = time.perf_counter() - t0

    # ---- cluster bootstrap replicates of arm means (shared by C1, C2a)
    t0 = time.perf_counter()
    W = r1.multinomial(T, np.full(T, 1 / T), size=(S, B)).astype(np.float32)  # (S, B, T)
    arm_star = np.einsum("sbt,sta->sba", W, tm.astype(np.float32)) / T  # (S, B, A)
    t_reps = time.perf_counter() - t0
    div_star, con_star = stats_from_arm_means(arm_star, D)
    res["C1 cluster pct"] = (pct(div_star).T, pct(con_star).T)
    timings["C1 cluster pct"] = time.perf_counter() - t0

    # ---- C2a: simultaneous max-|t| from the cluster replicates
    t0 = time.perf_counter()

    def simult(est, star):  # est (S, D), star (S, B, D)
        se = star.std(axis=1, ddof=1)
        safe = np.where(se > 0, se, 1.0)
        tt = np.where(se[:, None, :] > 0, np.abs(star - est[:, None, :]) / safe[:, None, :], 0.0)
        c = np.quantile(tt.max(axis=2), 1 - ALPHA, axis=1)
        L = est - c[:, None] * se
        U = est + c[:, None] * se
        return np.stack([L.max(axis=1), U.max(axis=1)], axis=1)

    est_g = arm[:, :D]
    est_c = arm[:, :D] - arm[:, D : D + 1]
    res["C2a simult. cluster max-|t|"] = (
        simult(est_g, arm_star[..., :D]),
        simult(est_c, arm_star[..., :D] - arm_star[..., D : D + 1]),
    )
    timings["C2a simult. cluster max-|t|"] = t_reps + time.perf_counter() - t0

    # ---- C2b: analytic Bonferroni-t on template-level means
    t0 = time.perf_counter()
    crit = tdist.ppf(1 - ALPHA / (2 * D), df=T - 1)

    def bonf(x):  # x (S, T, D) template-level values
        m = x.mean(axis=1)
        se = x.std(axis=1, ddof=1) / np.sqrt(T)
        return np.stack([(m - crit * se).max(axis=1), (m + crit * se).max(axis=1)], axis=1)

    res["C2b simult. Bonferroni-t"] = (bonf(tm[..., :D]), bonf(tm[..., :D] - tm[..., D : D + 1]))
    timings["C2b simult. Bonferroni-t"] = time.perf_counter() - t0

    # ---- C2c: m-out-of-n template bootstrap, root interval
    t0 = time.perf_counter()
    m = max(2, int(round(T**0.7)))
    Wm = r2.multinomial(m, np.full(T, 1 / T), size=(S, B)).astype(np.float32)
    arm_m = np.einsum("sbt,sta->sba", Wm, tm.astype(np.float32)) / m
    dm, cm = stats_from_arm_means(arm_m, D)
    scale = np.sqrt(m / T)

    def root(v, star):
        q_lo, q_hi = np.quantile(star - v[:, None], [ALPHA / 2, 1 - ALPHA / 2], axis=1)
        return np.stack([v - scale * q_hi, v - scale * q_lo], axis=1)

    res["C2c m-out-of-n cluster"] = (root(v_div, dm), root(v_con, cm))
    timings["C2c m-out-of-n cluster"] = time.perf_counter() - t0

    rows = []
    for meth, (ci_div, ci_con) in res.items():
        for metric, ci, truth, val in (
            ("divergence", ci_div, truth_div, v_div),
            ("contrast", ci_con, truth_con, v_con),
        ):
            lo, hi = ci[:, 0], ci[:, 1]
            cov = (lo <= truth + 1e-9) & (truth - 1e-9 <= hi)
            excl_pt = (val < lo - 1e-9) | (val > hi + 1e-9)
            excl0 = (lo > 1e-9) | (hi < -1e-9)
            rows.append(
                {
                    "T": T,
                    "D": D,
                    "rho": rho,
                    "delta": delta,
                    "metric": metric,
                    "method": meth,
                    "coverage": cov.mean(),
                    "coverage_mcse": np.sqrt(cov.mean() * (1 - cov.mean()) / S),
                    "width": float(np.mean(hi - lo)),
                    "excl_point": excl_pt.mean(),
                    "excl0": excl0.mean(),
                    "ms_per_dataset": 1000 * timings[meth] / S,
                }
            )
    return rows


def main():
    rows = []
    t_start = time.perf_counter()
    for T in TS:
        for D in DS:
            for rho in RHOS:
                for delta in DELTAS:
                    rows.extend(run_cell(T, D, rho, delta))
        print(f"T={T} done ({time.perf_counter() - t_start:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "part3b_llm_template_sim.csv", index=False)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
