"""Seed-to-seed Monte Carlo spread of today's LLM CI endpoints at B=200 vs B=2000.

Uses the hiring fixture's 27 pooled pair values (exactly what bootstrap_ci receives today).
"""

import json
from pathlib import Path

import numpy as np

from fairness_pipeline_dev_toolkit.llm_evals import run_llm_eval
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    expanded_recorded_counterfactual_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    generate_counterfactual_prompts,
    matched_pairwise_divergences,
    response_key,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci

cfg = expanded_recorded_counterfactual_config()
res = run_llm_eval(cfg, with_ci=False)
cf = cfg.counterfactual
prompts = generate_counterfactual_prompts(cf.template, cf.dimensions, cf.defaults, cf.name_pools)
rows = res.transcripts["counterfactual"]
responses = {response_key(p): r["response"] for p, r in zip(prompts, rows)}
x = np.asarray(matched_pairwise_divergences(prompts, responses))
out = {"n_pairs": int(x.size)}
for B in (200, 1000, 2000, 5000):
    ends = np.array([bootstrap_ci(x, np.mean, B=B, random_state=s) for s in range(200)])
    out[f"B={B}"] = {
        "lower_mean": float(ends[:, 0].mean()),
        "lower_sd_across_seeds": float(ends[:, 0].std(ddof=1)),
        "upper_sd_across_seeds": float(ends[:, 1].std(ddof=1)),
        "sd_as_share_of_width": float(
            ends.std(axis=0, ddof=1).mean() / np.mean(ends[:, 1] - ends[:, 0])
        ),
    }
(Path(__file__).parent / "results" / "b_stability.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
