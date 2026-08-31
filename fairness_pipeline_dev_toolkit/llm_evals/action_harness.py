"""Local harness for Action-shaped ``llm-fairness-check`` inputs.

``SvrusIO/fairpipe-action`` is a separate repository. This helper is what *this*
package exposes so that companion-repo can map ``with:`` keys onto
``fairpipe llm-eval`` without forking the gate. Wiring a real
``llm-fairness-check`` mode into the Action is BL-010.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_USAGE,
)

ActionInput = Union[str, float, bool, None]


def _as_optional_str(value: ActionInput) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: ActionInput, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def run_llm_fairness_check(inputs: Mapping[str, Any]) -> int:
    """Run ``fairpipe llm-eval`` from GitHub Action-shaped ``with:`` inputs.

    Expected keys (mirroring README ``fairness-check``, plus LLM-eval config):

    * ``config`` — path to an ``llm_eval`` YAML file (required)
    * ``metric`` — metric key to gate
    * ``threshold`` — acceptance band (string or float, as Action inputs are strings)
    * ``fail-on-violation`` — when false, a threshold *fail* (exit 1) becomes 0;
      usage (2) and illustrative (3) are unchanged
    """
    from fairness_pipeline_dev_toolkit.cli.main import main

    config = _as_optional_str(inputs.get("config"))
    if config is None:
        return EXIT_USAGE

    argv = ["llm-eval", "--config", config]
    metric = _as_optional_str(inputs.get("metric"))
    if metric is not None:
        argv.extend(["--metric", metric])
    threshold = _as_optional_str(inputs.get("threshold"))
    if threshold is not None:
        argv.extend(["--threshold", threshold])

    try:
        exit_code = main(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_PASS
        if isinstance(code, int):
            return code
        return EXIT_USAGE

    if exit_code is None:
        return EXIT_PASS
    if not _as_bool(inputs.get("fail-on-violation"), default=True) and exit_code == EXIT_FAIL:
        return EXIT_PASS
    return int(exit_code)
