from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from fairness_pipeline_dev_toolkit.exceptions import IntervalUndefinedError
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult

from .._async_utils import run_coroutine
from ..bbq import item_to_prompt, load_bbq_items, stereotypical_answer_index
from ..client import LLMClient
from ..config import LLMEvalConfig
from ..guards import DEFAULT_LLM_MIN_GROUP_SIZE, apply_min_group_size
from ..provenance import with_stereotype_caveat
from ..scoring import binary_rate_gap_interval, n_per_group, rate_disparity


def _parse_choice(text: str) -> Optional[int]:
    match = re.search(r"\b([012])\b", text or "")
    if match:
        return int(match.group(1))
    return None


class StereotypeAssociationEvaluator:
    """Experimental stereotyped-answer rate gap on BBQ-schema items (not Parrish BBQ).

    Every result carries an experimental ``caveat`` and therefore gates as
    illustrative until real BBQ bias scoring lands (issue #31).
    """

    name = "stereotype_association_score"

    def __init__(
        self,
        config: LLMEvalConfig,
        client: LLMClient,
        *,
        items: Optional[List[Dict[str, Any]]] = None,
        bbq_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.items = items
        self.bbq_path = bbq_path

    async def run_async(
        self,
        *,
        min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
        allow_small_samples: bool = False,
        with_ci: bool = True,
        ci_level: float = 0.95,
        bootstrap_B: int = 200,
        random_state: int = 42,
    ) -> tuple[MetricResult, List[Dict[str, str]]]:
        items = (
            self.items
            if self.items is not None
            else load_bbq_items(self.bbq_path or self.config.bbq_path)
        )
        prompts = [item_to_prompt(item) for item in items]
        texts = await self.client.complete_batch(prompts, params=self.config.params)
        rows: List[Dict[str, str]] = []
        scored: Dict[str, List[float]] = {}
        for item, prompt, text in zip(items, prompts, texts):
            group = str(item.get("group") or item.get("category") or "unknown")
            choice = _parse_choice(text)
            stereo = stereotypical_answer_index(item)
            hit = 1.0 if choice is not None and choice == stereo else 0.0
            scored.setdefault(group, []).append(hit)
            rows.append(
                {
                    "dimension": str(item.get("category") or "bbq"),
                    "group": group,
                    "prompt": prompt,
                    "response": text,
                }
            )

        counts = n_per_group(scored)
        eligible, can_compute = apply_min_group_size(
            counts, min_group_size, allow_small_samples=allow_small_samples
        )
        if not can_compute:
            return (
                with_stereotype_caveat(
                    MetricResult(
                        metric="stereotype_association_score",
                        value=float("nan"),
                        ci=None,
                        effect_size=float("nan"),
                        n_per_group=eligible,
                    )
                ),
                rows,
            )
        keep = set(eligible) if not allow_small_samples else set(counts)
        scores = {g: v for g, v in scored.items() if g in keep}
        value = rate_disparity(scores)
        ci = ci_kind = ci_note = None
        if with_ci and np.isfinite(value):
            _ = bootstrap_B
            _ = random_state
            try:
                ci = binary_rate_gap_interval(scores, level=ci_level)
                ci_kind = "simultaneous_pairwise"
            except IntervalUndefinedError as err:
                ci_note = err.ci_note
        reporting = counts if allow_small_samples else eligible
        return (
            with_stereotype_caveat(
                MetricResult(
                    metric="stereotype_association_score",
                    value=float(value),
                    ci=ci,
                    effect_size=float(value) if np.isfinite(value) else float("nan"),
                    n_per_group=reporting,
                    p_value=None,
                    ci_kind=ci_kind,
                    ci_note=ci_note,
                )
            ),
            rows,
        )

    def stereotype_association_score(self, **kwargs: Any) -> MetricResult:
        return run_coroutine(self.run_async(**kwargs))[0]
