"""Within-group lexical baseline for ``demographic_swap_divergence`` (BL-012).

Nine live Haiku responses: one asylum template × three same-coded names per group.
This is the no-effect baseline, not a group-effect measurement. Do not cite the
evaluator's matched-by-template pairing on this cache as a disparity finding.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_counterfactual import (
    RECORDED_MODEL,
    RECORDED_PROVIDER,
)
from fairness_pipeline_dev_toolkit.llm_evals.fixtures.recorded_group_rates import (
    HUMANITARIAN_REFUSAL_PARAMS,
    HUMANITARIAN_REFUSAL_TEMPLATES,
)

_ROOT = Path(__file__).resolve().parent / "recorded_within_group_control"
RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR = _ROOT / "cache"
RECORDED_WITHIN_GROUP_CONTROL_MANIFEST_PATH = _ROOT / "manifest.json"

WITHIN_GROUP_CONTROL_PROVIDER = RECORDED_PROVIDER
WITHIN_GROUP_CONTROL_MODEL = RECORDED_MODEL
WITHIN_GROUP_CONTROL_TEMPLATE = HUMANITARIAN_REFUSAL_TEMPLATES[0]
WITHIN_GROUP_CONTROL_PARAMS = dict(HUMANITARIAN_REFUSAL_PARAMS)
WITHIN_GROUP_CONTROL_NAMES: Dict[str, List[str]] = {
    "woman": ["Amina", "Fatima", "Leyla"],
    "man": ["Tariq", "Hassan", "Omar"],
    "ambiguous": ["Noor", "Kiran", "Alex"],
}


def load_recorded_within_group_control_manifest() -> Dict[str, Any]:
    from fairness_pipeline_dev_toolkit.llm_evals.fixtures._deprecation import (
        warn_fixture_helper_deprecated,
    )

    warn_fixture_helper_deprecated("load_recorded_within_group_control_manifest")
    if not RECORDED_WITHIN_GROUP_CONTROL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Within-group control manifest not found at "
            f"{RECORDED_WITHIN_GROUP_CONTROL_MANIFEST_PATH}."
        )
    return json.loads(RECORDED_WITHIN_GROUP_CONTROL_MANIFEST_PATH.read_text(encoding="utf-8"))


def load_within_group_control_records() -> List[Dict[str, str]]:
    """Return the nine ``{group, name, prompt, response}`` control recordings.

    .. deprecated::
        Will be removed in the next release (packaged fixture helper).
    """
    from fairness_pipeline_dev_toolkit.llm_evals.fixtures._deprecation import (
        warn_fixture_helper_deprecated,
    )

    warn_fixture_helper_deprecated("load_within_group_control_records")
    # Suppress the nested manifest helper warning — one FutureWarning per call site.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        manifest = load_recorded_within_group_control_manifest()
    records: List[Dict[str, str]] = []
    for entry in manifest["prompts"]:
        path = RECORDED_WITHIN_GROUP_CONTROL_CACHE_DIR / f"{entry['cache_key']}.txt"
        records.append(
            {
                "dimension": entry["dimension"],
                "group": entry["group"],
                "name": entry["name"],
                "prompt": entry["prompt"],
                "response": path.read_text(encoding="utf-8"),
            }
        )
    return records
