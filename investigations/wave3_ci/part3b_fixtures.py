"""Part 3b (fixtures) — current vs candidate CIs on the shipped recorded caches.

Replays committed caches only (assert-free here, but the kill switch is never set,
so a cache miss raises LiveLLMCallForbidden). Computes, per fixture:
  current   the value + CI the package returns today (B=200, seed 42),
  C1        template-cluster percentile bootstrap of the exact statistic,
  C2a       simultaneous cluster max-|t| inversion,
  C2b       simultaneous Bonferroni-t inversion on template-level means,
and the same for the two-dimension BL-016 fixture. B=1000 / 10000 for candidates.

Run: PYTHONPATH=.:investigations/wave3_ci .venv/bin/python investigations/wave3_ci/part3b_fixtures.py
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from part2_repro import FixedClient
from scipy.stats import t as tdist

from fairness_pipeline_dev_toolkit.llm_evals import (
    CounterfactualConfig,
    LLMEvalConfig,
    run_llm_eval,
)
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.demographic_swap import (
    DemographicSwapEvaluator,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    expanded_recorded_counterfactual_config,
    humanitarian_contrast_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_group_rates import (
    default_recorded_refusal_config,
    humanitarian_divergence_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    extract_response_features,
    generate_counterfactual_prompts,
    iter_matched_pairs,
    pairwise_divergence,
    response_key,
)
from fairness_pipeline_dev_toolkit.llm_evals.scoring import refusal_score

OUT = Path(__file__).resolve().parent / "results"
warnings.simplefilter("ignore", FutureWarning)
ALPHA = 0.05


def pair_table(prompts, responses):
    """{dimension: array (T, n_pairs)} of matched pair divergences, template-ordered."""
    by = defaultdict(lambda: defaultdict(list))
    for a, b in iter_matched_pairs(prompts):
        ta, tb = responses[response_key(a)], responses[response_key(b)]
        v = pairwise_divergence(
            extract_response_features(ta, reference=tb), extract_response_features(tb, reference=ta)
        )
        by[a.dimension][a.replicate_id].append(v)
    return {dim: np.array([v for _, v in sorted(t.items())]) for dim, t in by.items()}


def candidates(tm, gated, control, B, seed=7):
    """tm: {dim: (T,) template-level means}. Returns C1/C2a/C2b intervals."""
    T = len(next(iter(tm.values())))
    G = np.stack([tm[d] for d in gated], axis=1)  # (T, D)
    D = G.shape[1]
    X = G - tm[control][:, None] if control else G
    est = X.mean(axis=0)
    value = est.max()
    rng = np.random.default_rng(seed)
    W = rng.multinomial(T, np.full(T, 1 / T), size=B) / T  # (B, T)
    star = W @ X  # (B, D)
    c1 = np.percentile(star.max(axis=1), [2.5, 97.5])
    se = star.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, 1)
    tt = np.where(se > 0, np.abs(star - est) / safe, 0).max(axis=1)
    c = np.quantile(tt, 1 - ALPHA)
    c2a = [(est - c * se).max(), (est + c * se).max()]
    crit = tdist.ppf(1 - ALPHA / (2 * D), df=T - 1)
    se_t = X.std(axis=0, ddof=1) / np.sqrt(T)
    c2b = [(est - crit * se_t).max(), (est + crit * se_t).max()]
    return {
        "T": T,
        "value_recomputed": float(value),
        "C1": [float(x) for x in c1],
        "C2a": [float(x) for x in c2a],
        "C2b": [float(x) for x in c2b],
    }


def fixture(name, cfg, evaluator, control=None):
    res = run_llm_eval(cfg, with_ci=True, bootstrap_B=200, random_state=42)
    metric = res.metrics[evaluator]
    cf = cfg.counterfactual
    prompts = generate_counterfactual_prompts(
        cf.template, cf.dimensions, cf.defaults, cf.name_pools, control_dimension=control
    )
    rows = res.transcripts["counterfactual"]
    assert [r["prompt"] for r in rows] == [p.prompt for p in prompts]
    responses = {response_key(p): r["response"] for p, r in zip(prompts, rows)}
    tab = pair_table(prompts, responses)
    tm = {d: v.mean(axis=1) for d, v in tab.items()}
    gated = [d for d in tm if d != control]
    out = {
        "fixture": name,
        "metric": evaluator,
        "current_value": metric.value,
        "current_ci_B200": list(metric.ci),
    }
    out.update(candidates(tm, gated, control, B=10_000))
    out["template_level_means"] = {d: [round(float(x), 4) for x in v] for d, v in tm.items()}
    return out


def refusal_fixture():
    cfg = default_recorded_refusal_config()
    res = run_llm_eval(cfg, with_ci=True, bootstrap_B=200, random_state=42)
    m = res.metrics["refusal_rate_disparity"]
    rows = res.transcripts["refusal"]
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(refusal_score(r["response"]))
    return {
        "fixture": "recorded_refusal",
        "metric": "refusal_rate_disparity",
        "value": m.value,
        "current_ci_B200": list(m.ci),
        "per_group_refusals_by_template": {g: v for g, v in by.items()},
    }


def bl016_fixture():
    out = {}
    for kind in ("divergence", "contrast"):
        if kind == "divergence":
            tmpl = [f"{{gender}} {{age}} template{i}" for i in range(20)]
            dims = {"gender": ["A", "B"], "age": ["C", "D"]}
            control = None
        else:
            tmpl = [f"{{gender}} {{age}} {{control}} template{i}" for i in range(20)]
            dims = {"gender": ["A", "B"], "age": ["C", "D"], "control": ["E", "F"]}
            control = "control"
        cfg = LLMEvalConfig(
            provider="local",
            model="fixed",
            evaluators=["demographic_swap_divergence"],
            counterfactual=CounterfactualConfig(
                template=tmpl, dimensions=dims, control_dimension=control
            ),
        )
        ev = DemographicSwapEvaluator(cfg, FixedClient())
        prompts, responses, _ = asyncio.run(ev.prepare_async())
        tab = pair_table(prompts, responses)
        tm = {d: v.mean(axis=1) for d, v in tab.items()}
        gated = [d for d in tm if d != control]
        out[kind] = candidates(tm, gated, control, B=10_000)
    return out


def main():
    res = {
        "hiring_expanded_divergence": fixture(
            "recorded_counterfactual_expanded",
            expanded_recorded_counterfactual_config(),
            "demographic_swap_divergence",
        ),
        "humanitarian_divergence": fixture(
            "recorded_refusal (divergence replay)",
            humanitarian_divergence_config(),
            "demographic_swap_divergence",
        ),
        "humanitarian_contrast": fixture(
            "recorded_humanitarian_contrast",
            humanitarian_contrast_config(),
            "demographic_swap_contrast",
            control="control",
        ),
        "refusal": refusal_fixture(),
        "bl016_two_dimension": bl016_fixture(),
    }
    text = json.dumps(res, indent=1, default=float)
    (OUT / "part3b_fixtures.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
