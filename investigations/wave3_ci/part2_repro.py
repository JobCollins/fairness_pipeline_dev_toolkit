"""Part 2 — deterministic reproductions on current main (package code untouched).

BL-016  two-dimension fixture: divergence and contrast CIs vs the reported point.
BL-014  analyzer DPD percentile CI coverage of 0 at equality (3 groups, n=100, p=0.5),
        and refusal_rate_disparity's CI with equal true refusal rates.
BL-031  analyzer ci_method="bca" with a small minority group -> NaN replicates.

Run: PYTHONPATH=. .venv/bin/python investigations/wave3_ci/part2_repro.py
"""

from __future__ import annotations

import asyncio
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

from fairness_pipeline_dev_toolkit.llm_evals import CounterfactualConfig, LLMEvalConfig
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.demographic_swap import (
    DemographicSwapEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.refusal import (
    RefusalRateEvaluator,
)
from fairness_pipeline_dev_toolkit.metrics.core import (
    FairnessAnalyzer,
    dpd_stat_from_indices,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
warnings.simplefilter("ignore", FutureWarning)


def mcse(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n)


# --------------------------------------------------------------------------- BL-016
class FixedClient:
    """Offline client: deterministic text from the prompt. No network."""

    provider = "local"
    model = "fixed"

    def available(self) -> bool:
        return True

    async def complete_batch(self, prompts, params=None):
        return [
            (
                "I cannot recommend unsuitable poor"
                if p.startswith("B ")
                else "excellent strong qualified recommend"
            )
            for p in prompts
        ]


def bl016() -> dict:
    templates_div = [f"{{gender}} {{age}} template{i}" for i in range(20)]
    cfg = LLMEvalConfig(
        provider="local",
        model="fixed",
        evaluators=["demographic_swap_divergence"],
        counterfactual=CounterfactualConfig(
            template=templates_div, dimensions={"gender": ["A", "B"], "age": ["C", "D"]}
        ),
    )
    ev = DemographicSwapEvaluator(cfg, FixedClient())
    div = asyncio.run(ev.run_async())[0]

    templates_con = [f"{{gender}} {{age}} {{control}} template{i}" for i in range(20)]
    cfg_c = LLMEvalConfig(
        provider="local",
        model="fixed",
        evaluators=["demographic_swap_contrast"],
        counterfactual=CounterfactualConfig(
            template=templates_con,
            dimensions={"gender": ["A", "B"], "age": ["C", "D"], "control": ["E", "F"]},
            control_dimension="control",
        ),
    )
    ev_c = DemographicSwapEvaluator(cfg_c, FixedClient())
    con = asyncio.run(ev_c.run_contrast_async())[0]

    out = {
        "divergence": {
            "value": div.value,
            "ci": list(div.ci),
            "ci_excludes_point": not (div.ci[0] <= div.value <= div.ci[1]),
        },
        "contrast": {
            "value": con.value,
            "ci": list(con.ci),
            "ci_excludes_point": not (con.ci[0] <= con.value <= con.ci[1]),
        },
        "note": "20 templates, each template yields identical responses, so any template-cluster "
        "bootstrap that recomputes max-over-dimensions returns the degenerate CI [value, value].",
    }
    return out


# --------------------------------------------------------------------------- BL-014
def bl014_analyzer(n_sim: int = 500, B: int = 1000) -> dict:
    fa = FairnessAnalyzer(min_group_size=30, backend="native")
    s = np.repeat(["A", "B", "C"], 100)
    covers, lows = 0, []
    for seed in range(n_sim):
        rng = np.random.default_rng(10_000 + seed)
        y = rng.binomial(1, 0.5, 300)
        r = fa.demographic_parity_difference(y, s, ci_samples=B, with_effect_size=False)
        lo, hi = r.ci
        covers += lo <= 0.0 <= hi
        lows.append(lo)
    cov = covers / n_sim
    return {
        "setting": "3 groups x n=100, Bernoulli(0.5), percentile, B=%d, %d sims" % (B, n_sim),
        "coverage_of_0": cov,
        "mcse": mcse(cov, n_sim),
        "lower_bound_min": float(np.min(lows)),
        "lower_bound_median": float(np.median(lows)),
        "lower_bound_max": float(np.max(lows)),
    }


class BernoulliRefusalClient:
    """Offline client: each prompt refuses with probability p, seeded per run."""

    provider = "local"
    model = "bernoulli"

    def __init__(self, p: float, seed: int) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)

    def available(self) -> bool:
        return True

    async def complete_batch(self, prompts, params=None):
        draws = self.rng.random(len(prompts)) < self.p
        return ["I cannot help with that." if d else "Approved." for d in draws]


def bl014_refusal(n_sim: int = 500, B: int = 1000) -> dict:
    out = {}
    for T in (5, 30, 100):
        cfg = LLMEvalConfig(
            provider="local",
            model="bernoulli",
            evaluators=["refusal_rate_disparity"],
            counterfactual=CounterfactualConfig(
                template=[f"Case {{gender}} number {i}." for i in range(T)],
                dimensions={"gender": ["woman", "man", "ambiguous"]},
            ),
        )
        covers = undefined = 0
        for seed in range(n_sim):
            ev = RefusalRateEvaluator(cfg, BernoulliRefusalClient(0.3, 20_000 + seed))
            m = asyncio.run(ev.run_async(bootstrap_B=B))[0]
            if m.ci is None or not np.isfinite(m.value):
                undefined += 1
                continue
            covers += m.ci[0] <= 0.0 <= m.ci[1]
        n_def = n_sim - undefined
        cov = covers / n_def
        out[f"T={T}"] = {
            "templates_per_group": T,
            "true_rates": "all 0.3",
            "coverage_of_0": cov,
            "mcse": mcse(cov, n_def),
            "undefined": undefined,
        }
    return out


# --------------------------------------------------------------------------- BL-031
def bl031() -> dict:
    """Small minority group with min_group_size lowered so it is analysed."""
    results = {}
    for m_minor in (1, 2, 3, 5):
        rng = np.random.default_rng(31)
        n_major = 300
        y = np.concatenate([rng.binomial(1, 0.5, n_major), rng.binomial(1, 0.5, m_minor)])
        s = np.array(["maj"] * n_major + ["min"] * m_minor)
        fa = FairnessAnalyzer(min_group_size=1, backend="native")
        row: dict = {"n_major": n_major, "n_minor": m_minor}
        B = 1000
        try:
            r = fa.demographic_parity_difference(
                y, s, ci_method="bca", ci_samples=B, with_effect_size=False
            )
            row["value"] = r.value
            row["bca_ci_package"] = list(r.ci)
        except Exception as exc:  # report the actual failure mode
            row["bca_ci_package"] = f"{type(exc).__name__}: {exc}"
        r_pct = fa.demographic_parity_difference(
            y, s, ci_method="percentile", ci_samples=B, with_effect_size=False
        )
        row["percentile_ci_package"] = list(r_pct.ci)

        # Reconstruct the exact replicate vector bootstrap_ci produced (same seed, same loop).
        groups = ["maj", "min"]
        group_of = s.astype(str)
        x = np.arange(len(y))
        bg = np.random.default_rng(42)
        boot = np.empty(B)
        for b in range(B):
            boot[b] = dpd_stat_from_indices(x[bg.integers(0, len(x), len(x))], y, group_of, groups)
        theta = dpd_stat_from_indices(x, y, group_of, groups)
        nan_share = float(np.mean(~np.isfinite(boot)))
        p_less_pkg = float(np.mean(boot < theta))  # NaN compares False -> counted as "not less"
        fin = boot[np.isfinite(boot)]
        p_less_fin = float(np.mean(fin < theta))
        jack = np.array(
            [dpd_stat_from_indices(np.delete(x, i), y, group_of, groups) for i in range(len(x))]
        )
        row.update(
            {
                "nan_replicate_share": nan_share,
                "theoretical_nan_share": (1 - m_minor / len(y)) ** len(y),
                "z0_package (NaN counted as not-less)": float(
                    norm.ppf(np.clip(p_less_pkg, 1e-6, 1 - 1e-6))
                ),
                "z0_finite_only": float(norm.ppf(np.clip(p_less_fin, 1e-6, 1 - 1e-6))),
                "jackknife_nan_count": int(np.sum(~np.isfinite(jack))),
                "np.percentile_on_boot_with_nan": float(np.percentile(boot, 50)),
                "fallback_guard_fires": bool(len(x) < 5 or np.any(~np.isfinite(x))),
            }
        )
        results[f"minor={m_minor}"] = row
    return results


def main() -> None:
    res = {"BL-016": bl016()}
    print(json.dumps(res["BL-016"], indent=1))
    res["BL-031"] = bl031()
    print(json.dumps(res["BL-031"], indent=1, default=str))
    res["BL-014_analyzer"] = bl014_analyzer()
    print(json.dumps(res["BL-014_analyzer"], indent=1))
    res["BL-014_refusal"] = bl014_refusal()
    print(json.dumps(res["BL-014_refusal"], indent=1))
    (OUT / "part2_repro.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
