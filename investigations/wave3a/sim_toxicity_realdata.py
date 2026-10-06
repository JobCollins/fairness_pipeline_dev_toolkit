"""Wave 3a PR B — real-data check for toxicity paired-t (decision 8).

Same design as ``sim_tmin_c2b.py``: treat the recorded toxicity fixture's
per-template group scores as a finite population, centre each group, add an
effect to one group, resample templates at T ∈ {3,5,7,9}, and measure paired
Bonferroni-t coverage of the true max−min gap.

Run:  .venv/bin/python investigations/wave3a/sim_toxicity_realdata.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.llm_evals._async_utils import run_coroutine
from fairness_pipeline_dev_toolkit.llm_evals.cache import ResponseCache
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_toxicity_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    generate_counterfactual_prompts,
)
from fairness_pipeline_dev_toolkit.llm_evals.runner import build_client
from fairness_pipeline_dev_toolkit.llm_evals.scoring import toxicity_score
from fairness_pipeline_dev_toolkit.stats.gap_intervals import simultaneous_gap_bounds

OUT = Path(__file__).resolve().parent / "results"
TS = (3, 5, 7, 9)
EFFECTS = (0.0, 0.05, 0.10)
S = 2000
TOL = 1e-12


def _load_population() -> tuple[np.ndarray, list[str]]:
    """Return (T, K) score matrix and group names from the recorded toxicity fixture."""
    cfg = default_recorded_toxicity_config()
    cache = ResponseCache(cfg.cache_dir) if cfg.cache_dir else None
    client = build_client(cfg, cache=cache, replay_only=True)
    prompts = generate_counterfactual_prompts(
        cfg.counterfactual.template,
        cfg.counterfactual.dimensions,
        cfg.counterfactual.defaults,
        cfg.counterfactual.name_pools,
    )
    texts = run_coroutine(client.complete_batch([p.prompt for p in prompts], params=cfg.params))
    by_group: dict[str, dict] = {}
    for item, text in zip(prompts, texts):
        by_group.setdefault(item.group, {})[item.replicate_id] = toxicity_score(text)
    groups = sorted(by_group)
    complete = set.intersection(*(set(by_group[g]) for g in groups))
    reps = sorted(complete)
    mat = np.column_stack([[by_group[g][r] for r in reps] for g in groups])
    return mat, groups


def _paired_t(mat: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Raw paired Bonferroni-t (no package floor); needs T ≥ 2."""
    T, k = mat.shape
    if T < 2:
        raise IntervalUndefinedError("too_few_templates", f"T={T} < 2")
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    a_tail = (1.0 - level) / (2.0 * len(pairs))
    crit = float(student_t.ppf(1.0 - a_tail, T - 1))
    lo, hi = [], []
    for i, j in pairs:
        d = mat[:, i] - mat[:, j]
        sd = float(d.std(ddof=1))
        if sd <= 0.0:
            raise IntervalUndefinedError("zero_variance", "all template diffs equal")
        se = sd / np.sqrt(T)
        m = float(d.mean())
        lo.append(m - crit * se)
        hi.append(m + crit * se)
    return simultaneous_gap_bounds(lo, hi, max_gap=1.0)


def _coverage(pop: np.ndarray, T: int, effect: float, s: int, rng) -> dict:
    N, k = pop.shape
    centred = pop - pop.mean(axis=0, keepdims=True)
    true_mat = centred.copy()
    true_mat[:, 0] = true_mat[:, 0] + effect
    true_gap = float(true_mat.mean(axis=0).max() - true_mat.mean(axis=0).min())
    cov = excl = undef = 0
    for _ in range(s):
        idx = rng.integers(0, N, T)
        sample = true_mat[idx]
        point = float(sample.mean(axis=0).max() - sample.mean(axis=0).min())
        try:
            lo, hi = _paired_t(sample, level=0.95)
            lo, hi = min(lo, point), max(hi, point)
        except IntervalUndefinedError:
            undef += 1
            continue
        cov += lo - TOL <= true_gap <= hi + TOL
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
    pop, groups = _load_population()
    print("population", pop.shape, "groups", groups, "var", float(pop.var()))
    rows = []
    for T in TS:
        for effect in EFFECTS:
            rng = np.random.default_rng([3, T, int(effect * 100), 13])
            row = _coverage(pop, T, effect, S, rng)
            rows.append(row)
            print(row)

    by_T = {}
    for T in TS:
        gap0 = [r for r in rows if r["T"] == T and r["effect"] == 0.0]
        mean_c = float(np.mean([r["coverage"] for r in gap0]))
        min_c = float(np.min([r["coverage"] for r in gap0]))
        excl = float(np.max([r["excl_point"] for r in gap0]))
        undef = float(np.max([r["undefined"] for r in gap0]))
        by_T[str(T)] = {
            "mean": mean_c,
            "min": min_c,
            "excl_point_max": excl,
            "undefined_max": undef,
            "pass": bool(mean_c >= 0.95 and min_c >= 0.93 and excl == 0.0),
        }
    passing = [int(T) for T, v in by_T.items() if v["pass"]]
    summary = {
        "by_T": by_T,
        "pass_decision8": bool(passing) and all(v["pass"] for v in by_T.values()),
        "S": S,
        "TS": list(TS),
        "population": "recorded_toxicity",
        "native_T": int(pop.shape[0]),
        "rows": rows,
    }
    # Decision 8 for the default requires every evaluated T to pass (same bar as C2b).
    (OUT / "toxicity_realdata_summary.json").write_text(
        json.dumps(summary, indent=1, default=float)
    )
    print("SUMMARY", json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
