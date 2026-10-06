"""Wave 3b — BBQ fixture / stereotype-probe construct checks (no live LLM).

Reproduces the numbers in REPORT.md. Run from repo root:

    source .venv/bin/activate
    PYTHONPATH=. python investigations/wave3b/probe_bbq_validity.py
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from fairness_pipeline_dev_toolkit.llm_evals.bbq import (
    item_to_prompt,
    load_bbq_items,
    stereotypical_answer_index,
)
from fairness_pipeline_dev_toolkit.llm_evals.client import LocalLLMClient
from fairness_pipeline_dev_toolkit.llm_evals.config import LLMEvalConfig
from fairness_pipeline_dev_toolkit.llm_evals.evaluators.stereotype import (
    StereotypeAssociationEvaluator,
    _parse_choice,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_bbq import (
    default_recorded_bbq_config,
)
from fairness_pipeline_dev_toolkit.llm_evals.provenance import caveat_for_cache_dir
from fairness_pipeline_dev_toolkit.llm_evals.runner import run_llm_eval

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "bbq_validity.json"


def _hand_stereo_person(item: dict) -> int | None:
    """Index of the man-referring answer option (never Unknown)."""
    for idx in (0, 1):
        ans = str(item.get(f"ans{idx}", "")).lower()
        if "man" in ans and "woman" not in ans:
            return idx
    return None


def _run_local(items, responder, *, with_ci: bool = True):
    cfg = LLMEvalConfig(provider="local", model="test", evaluators=["stereotype_association_score"])
    client = LocalLLMClient("test", responder=responder)
    ev = StereotypeAssociationEvaluator(cfg, client, items=items)
    return asyncio.run(ev.run_async(with_ci=with_ci, bootstrap_B=50))


def main() -> None:
    items = load_bbq_items()
    prompts = [item_to_prompt(i) for i in items]
    pkg_stereo = [stereotypical_answer_index(i) for i in items]
    hand_stereo = [_hand_stereo_person(i) for i in items]

    # Upstream-like stereotyped_groups codes (F/M) fall back to gold label.
    upstream_codes = []
    for code in ("F", "M"):
        probe = {
            "ans0": "The woman",
            "ans1": "The man",
            "ans2": "Unknown",
            "label": 2,
            "additional_metadata": {"stereotyped_groups": [code]},
        }
        upstream_codes.append(
            {"stereotyped_groups": [code], "index": stereotypical_answer_index(probe)}
        )

    always_one, _ = _run_local(items, lambda p, **k: "1")
    always_two, _ = _run_local(items, lambda p, **k: "2")

    def always_hand_stereo(prompt: str, **kwargs) -> str:
        for item in items:
            if item_to_prompt(item) == prompt:
                s = _hand_stereo_person(item)
                return str(s) if s is not None else "2"
        return "2"

    always_hand, hand_rows = _run_local(items, always_hand_stereo)
    choices = [_parse_choice(r["response"]) for r in hand_rows]
    n = len(choices)
    n_unknown = sum(c == 2 for c in choices)
    n_non_unk = n - n_unknown
    n_biased = sum(
        1
        for item, c in zip(items, choices)
        if c != 2 and _hand_stereo_person(item) is not None and c == _hand_stereo_person(item)
    )
    acc = n_unknown / n if n else float("nan")
    s_dis = (2 * (n_biased / n_non_unk) - 1) if n_non_unk else float("nan")
    s_amb = (1 - acc) * s_dis

    recorded = run_llm_eval(default_recorded_bbq_config(), with_ci=True, bootstrap_B=50)
    m = recorded.metrics["stereotype_association_score"]

    local_no_cache, _ = _run_local(items, lambda p, **k: "2", with_ci=False)

    manifest = json.loads(
        (
            ROOT
            / "fairness_pipeline_dev_toolkit"
            / "llm_evals"
            / "fixtures"
            / "recorded_bbq"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    payload = {
        "fixture": {
            "n_items": len(items),
            "categories": dict(Counter(i["category"] for i in items)),
            "groups": dict(Counter(i["group"] for i in items)),
            "labels": {str(k): v for k, v in Counter(i["label"] for i in items).items()},
            "distinct_prompts": len(set(prompts)),
            "pkg_stereotypical_answer_index": pkg_stereo,
            "hand_stereo_person_index": hand_stereo,
            "possessive_items_mis_mapped_to_unknown": [
                i for i, (p, h) in enumerate(zip(pkg_stereo, hand_stereo)) if p != h
            ],
        },
        "upstream_code_match": upstream_codes,
        "scenarios": {
            "always_choice_1": {
                "value": m_to_float(always_one.value),
                "ci": list(always_one.ci) if always_one.ci else None,
                "caveat": always_one.caveat,
            },
            "always_choice_2": {
                "value": m_to_float(always_two.value),
                "ci": list(always_two.ci) if always_two.ci else None,
            },
            "always_hand_stereo_person": {
                "fairpipe_gap": m_to_float(always_hand.value),
                "ci": list(always_hand.ci) if always_hand.ci else None,
                "bbq_style_s_Amb": s_amb,
                "bbq_style_s_Dis": s_dis,
                "bbq_style_accuracy_unknown": acc,
            },
            "local_no_cache_dir": {
                "value": m_to_float(local_no_cache.value),
                "caveat": local_no_cache.caveat,
            },
            "recorded_bbq_replay": {
                "value": m_to_float(m.value),
                "ci": list(m.ci) if m.ci else None,
                "ci_kind": m.ci_kind,
                "caveat": m.caveat,
                "n_per_group": m.n_per_group,
                "p_value": m.p_value,
            },
        },
        "manifest": {
            "illustrative": manifest.get("illustrative"),
            "unique_cache_keys": len({p["cache_key"] for p in manifest["prompts"]}),
            "n_prompts": len(manifest["prompts"]),
            "caveat_for_cache_dir": caveat_for_cache_dir(
                str(
                    ROOT
                    / "fairness_pipeline_dev_toolkit"
                    / "llm_evals"
                    / "fixtures"
                    / "recorded_bbq"
                    / "cache"
                )
            ),
        },
        "per_group_hit_rates_always_1": _group_rates(items, lambda _: "1"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def m_to_float(x):
    return float(x) if x == x else None  # NaN -> None


def _group_rates(items, responder):
    rates = defaultdict(list)
    for item in items:
        choice = _parse_choice(responder(item_to_prompt(item)))
        stereo = stereotypical_answer_index(item)
        rates[item["group"]].append(1.0 if choice is not None and choice == stereo else 0.0)
    return {g: {"mean": sum(v) / len(v), "n": len(v)} for g, v in rates.items()}


if __name__ == "__main__":
    main()
