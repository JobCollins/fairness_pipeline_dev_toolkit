"""Wave 3a PR B — T_min evidence for C2b on recorded fixture template means.

Treats the shipped recorded fixtures as a finite population of template-level
arm values (loaded through ``llm_evals.fixtures`` helpers so the path survives
Wave 4 moves). Per arm the values are centred, a chosen effect is added to one
arm, templates are resampled at T ∈ {3,4,5,7,10}, and **raw** C2b coverage of
the true max-mean (or max contrast) is measured — including T=3 and T=4, which
the package helper refuses (``T_MIN_TEMPLATES=5``).

Run:  .venv/bin/python investigations/wave3a/sim_tmin_c2b.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fairness_pipeline_dev_toolkit.llm_evals._async_utils import run_coroutine
from fairness_pipeline_dev_toolkit.llm_evals.cache import ResponseCache
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.demographic_swap import (
    DemographicSwapEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    expanded_recorded_counterfactual_config,
    humanitarian_contrast_config,
    humanitarian_divergence_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.runner import build_client
from fairness_pipeline_dev_toolkit.llm_evals.template_ci import (
    contrast_template_arms,
    divergence_template_arms,
)

OUT = Path(__file__).resolve().parent / "results"
TS = (3, 4, 5, 7, 10)
EFFECTS = (0.0, 0.05, 0.10)
S = 2000
TOL = 1e-12


def _raw_c2b(arm_values: dict, level: float = 0.95) -> tuple[float, float]:
    """C2b without the package ``T_MIN_TEMPLATES`` floor (needs T ≥ 2 for df)."""
    arrays = {k: np.asarray(v, float) for k, v in arm_values.items()}
    T = next(iter(arrays.values())).size
    if T < 2:
        raise ValueError("need T >= 2")
    D = len(arrays)
    a_tail = (1.0 - level) / (2.0 * D)
    crit = float(student_t.ppf(1.0 - a_tail, T - 1))
    lowers, uppers = [], []
    all_zero = True
    for v in arrays.values():
        mean = float(v.mean())
        sd = float(v.std(ddof=1))
        if sd > 0.0:
            all_zero = False
            se = sd / np.sqrt(T)
            lowers.append(mean - crit * se)
            uppers.append(mean + crit * se)
        else:
            lowers.append(mean)
            uppers.append(mean)
    if all_zero:
        m = max(lowers)
        return m, m
    return float(max(lowers)), float(max(uppers))


def _load_arms(cfg_fn, kind: str):
    cfg = cfg_fn()
    cache = ResponseCache(cfg.cache_dir) if cfg.cache_dir else None
    client = build_client(cfg, cache=cache, replay_only=True)
    ev = DemographicSwapEvaluator(cfg, client)
    prompts, responses, _ = run_coroutine(ev.prepare_async())
    control = cfg.counterfactual.control_dimension
    if kind == "contrast":
        gated = [d for d in cfg.counterfactual.dimensions if d != control]
        return contrast_template_arms(
            prompts, responses, gated_dimensions=gated, control_dimension=control
        )
    dims = [d for d in cfg.counterfactual.dimensions if d != control]
    return divergence_template_arms(prompts, responses, dimensions=dims)


def _coverage(pop_arms: dict, T: int, effect: float, s: int, rng):
    names = list(pop_arms.keys())
    pop = {k: np.asarray(v, float) for k, v in pop_arms.items()}
    N = next(iter(pop.values())).size
    centred = {k: v - v.mean() for k, v in pop.items()}
    true_arms = {k: centred[k].copy() for k in names}
    true_arms[names[0]] = true_arms[names[0]] + effect
    true = max(float(v.mean()) for v in true_arms.values())
    cov = excl = undef = 0
    for _ in range(s):
        idx = rng.integers(0, N, T)
        sample = {k: true_arms[k][idx] for k in names}
        point = max(float(v.mean()) for v in sample.values())
        try:
            lo, hi = _raw_c2b(sample, level=0.95)
            lo, hi = min(lo, point), max(hi, point)
        except Exception:
            undef += 1
            continue
        cov += lo - TOL <= true <= hi + TOL
        excl += not (lo - TOL <= point <= hi + TOL)
    return {
        "T": T,
        "effect": effect,
        "coverage": cov / s,
        "excl_point": excl / s,
        "undefined": undef / s,
        "mcse": float(np.sqrt((cov / s) * (1 - cov / s) / s)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    populations = [
        ("divergence_expanded", expanded_recorded_counterfactual_config, "divergence"),
        ("divergence_humanitarian", humanitarian_divergence_config, "divergence"),
        ("contrast_humanitarian", humanitarian_contrast_config, "contrast"),
    ]
    for label, cfg_fn, kind in populations:
        arms, T_nat = _load_arms(cfg_fn, kind)
        print(label, "native_T", T_nat, "arms", {k: len(v) for k, v in arms.items()})
        if T_nat < 3 or not arms:
            continue
        for T in TS:
            for effect in EFFECTS:
                rng = np.random.default_rng([abs(hash(label)) % 10_000, T, int(effect * 100), 7])
                row = _coverage(arms, T, effect, S, rng)
                row.update(population=label, kind=kind, native_T=T_nat)
                rows.append(row)
                print(row)

    by_T = {}
    for T in TS:
        gap0 = [r for r in rows if r["T"] == T and r["effect"] == 0.0]
        mean_c = float(np.mean([r["coverage"] for r in gap0]))
        min_c = float(np.min([r["coverage"] for r in gap0]))
        excl = float(np.max([r["excl_point"] for r in gap0]))
        by_T[str(T)] = {
            "mean": mean_c,
            "min": min_c,
            "excl_point_max": excl,
            "pass": bool(mean_c >= 0.95 and min_c >= 0.93 and excl == 0.0),
        }
    passing = [int(T) for T, v in by_T.items() if v["pass"] and int(T) >= 3]
    T_min = min(passing) if passing else None
    summary = {"by_T": by_T, "T_min": T_min, "S": S, "raw_c2b": True, "rows": rows}
    (OUT / "tmin_c2b_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("SUMMARY", json.dumps({"by_T": by_T, "T_min": T_min}, indent=1))


if __name__ == "__main__":
    main()
