"""Part 1.3 — LLM cluster structure of the shipped fixtures (no live calls).

For every shipped counterfactual-style fixture config this prints:
- templates (= replicate_id clusters) per dimension and groups per dimension,
- matched pairs per (dimension, template) bucket and how many pairs reuse each response,
- byte-identical prompts that map to *different* response_keys (cross-arm sharing),
- for the BBQ fixture, how many items share one prompt/response.

Run: .venv/bin/python investigations/wave3_ci/part1_llm_cluster_structure.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from fairness_pipeline_dev_toolkit.llm_evals.cache import make_cache_key
from fairness_pipeline_dev_toolkit.llm_evals.fixtures import (
    default_recorded_counterfactual_config,
    expanded_recorded_counterfactual_config,
    humanitarian_contrast_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_bbq import (
    default_recorded_bbq_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_group_rates import (
    default_recorded_refusal_config,
    default_recorded_toxicity_config,
    humanitarian_divergence_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.probes.counterfactual import (
    generate_counterfactual_prompts,
    iter_matched_pairs,
    response_key,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def describe(name: str, cfg, evaluator: str) -> dict:
    cf = cfg.counterfactual
    control = cf.control_dimension if evaluator == "demographic_swap_contrast" else None
    prompts = generate_counterfactual_prompts(
        cf.template, cf.dimensions, cf.defaults, cf.name_pools, control_dimension=control
    )
    per_dim = defaultdict(lambda: {"templates": set(), "groups": set(), "prompts": 0})
    for p in prompts:
        d = per_dim[p.dimension]
        d["templates"].add(p.replicate_id)
        d["groups"].add(p.group)
        d["prompts"] += 1

    pairs_per_bucket = Counter()
    uses_per_response = Counter()
    for a, b in iter_matched_pairs(prompts):
        pairs_per_bucket[(a.dimension, a.replicate_id)] += 1
        uses_per_response[response_key(a)] += 1
        uses_per_response[response_key(b)] += 1

    by_cache = defaultdict(list)
    for p in prompts:
        by_cache[make_cache_key(cfg.provider, cfg.model, p.prompt, cfg.params)].append(
            response_key(p)
        )
    shared = {k: v for k, v in by_cache.items() if len(v) > 1}

    return {
        "fixture": name,
        "evaluator": evaluator,
        "n_prompts": len(prompts),
        "dimensions": {
            dim: {
                "templates_T": len(v["templates"]),
                "groups": sorted(v["groups"]),
                "prompts": v["prompts"],
            }
            for dim, v in per_dim.items()
        },
        "pairs_per_dim_template_bucket": sorted(set(pairs_per_bucket.values())),
        "pairs_reusing_each_response": sorted(set(uses_per_response.values())),
        "byte_identical_prompts_across_response_keys": [v for v in shared.values()],
    }


def bbq_structure() -> dict:
    cfg = default_recorded_bbq_config()
    from fairness_pipeline_dev_toolkit.llm_evals.bbq import (
        item_to_prompt,
        load_bbq_items,
    )

    items = load_bbq_items(cfg.bbq_path)
    prompts = [item_to_prompt(i) for i in items]
    groups = Counter(str(i.get("group") or i.get("category")) for i in items)
    share = Counter(prompts)
    return {
        "fixture": "recorded_bbq",
        "evaluator": "stereotype_association_score",
        "n_items": len(items),
        "items_per_group": dict(groups),
        "distinct_prompts": len(share),
        "items_per_distinct_prompt": sorted(set(share.values())),
    }


def main() -> None:
    rows = [
        describe(
            "recorded_counterfactual (n=1)",
            default_recorded_counterfactual_config(),
            "demographic_swap_divergence",
        ),
        describe(
            "recorded_counterfactual_expanded (hiring)",
            expanded_recorded_counterfactual_config(),
            "demographic_swap_divergence",
        ),
        describe(
            "recorded_toxicity (hiring copy)",
            default_recorded_toxicity_config(),
            "toxicity_sentiment_disparity",
        ),
        describe(
            "recorded_refusal (humanitarian)",
            default_recorded_refusal_config(),
            "refusal_rate_disparity",
        ),
        describe(
            "humanitarian_divergence (refusal cache)",
            humanitarian_divergence_config(),
            "demographic_swap_divergence",
        ),
        describe(
            "recorded_humanitarian_contrast",
            humanitarian_contrast_config(),
            "demographic_swap_contrast",
        ),
    ]
    rows.append(bbq_structure())
    text = json.dumps(rows, indent=1, default=list)
    (OUT / "part1_llm_cluster_structure.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
