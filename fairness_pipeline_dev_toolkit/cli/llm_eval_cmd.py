from __future__ import annotations

import argparse
import sys

from fairness_pipeline_dev_toolkit.llm_evals.client import (
    CacheMissError,
    LiveLLMCallForbidden,
)
from fairness_pipeline_dev_toolkit.llm_evals.config import load_llm_eval_config
from fairness_pipeline_dev_toolkit.llm_evals.gating import (
    EXIT_USAGE,
    GATE_STATUS_TO_EXIT,
    evaluate_llm_eval_gate,
)
from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE
from fairness_pipeline_dev_toolkit.llm_evals.runner import (
    results_to_markdown,
    run_llm_eval,
    write_transcripts,
)


def _selected_metric(args: argparse.Namespace) -> str | None:
    if args.metric is None:
        return None
    text = str(args.metric).strip()
    return text or None


def cmd_llm_eval(args: argparse.Namespace) -> int:
    metric = _selected_metric(args)
    if args.threshold is not None and metric is None:
        print(
            "error: --metric is required when --threshold is set",
            file=sys.stderr,
        )
        return EXIT_USAGE

    config = load_llm_eval_config(path=args.config)
    if metric is not None and metric not in config.evaluators:
        print(
            f"error: metric {metric!r} is not in this run's evaluators "
            f"{list(config.evaluators)}.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    from fairness_pipeline_dev_toolkit.llm_evals.runner import build_client

    client = build_client(config) if config.provider == "local" and not args.dry_run else None
    try:
        result = run_llm_eval(
            config,
            client=client,
            dry_run=args.dry_run,
            min_group_size=args.min_group_size,
            allow_small_samples=args.allow_small_samples,
            with_ci=args.with_ci,
            ci_level=args.ci_level,
            bootstrap_B=args.bootstrap_B,
            random_state=args.random_state,
        )
    except (CacheMissError, LiveLLMCallForbidden) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.dry_run:
        md = result.dry_run.to_markdown() if result.dry_run else ""
        print(md)
        report_path = args.report_md or args.out
        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)
        return 0

    md = results_to_markdown(result, title="LLM Fairness Evaluation Report")
    print(md)

    report_path = args.report_md or args.out
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)

    if args.transcripts_out and result.transcripts:
        write_transcripts(args.transcripts_out, result.transcripts)

    try:
        gate_status, _passed = evaluate_llm_eval_gate(
            result.metrics, threshold=args.threshold, metric=metric
        )
    except KeyError:
        print(
            f"error: metric {metric!r} was not present in the eval results.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    return GATE_STATUS_TO_EXIT[gate_status]


def register_llm_eval_parser(sub) -> None:
    p = sub.add_parser(
        "llm-eval",
        help="Run LLM fairness evaluation from an llm_eval YAML config",
    )
    p.add_argument("--config", required=True, help="Path to llm_eval YAML config")
    p.add_argument(
        "--report-md",
        help="Write Markdown report to this path (alias: --out)",
    )
    p.add_argument("--out", help="Alias for --report-md")
    p.add_argument(
        "--transcripts-out",
        help="Write raw probe transcripts JSON to this path (not rendered in report)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate request count and approximate cost without live provider calls",
    )
    p.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_LLM_MIN_GROUP_SIZE,
        help=(
            "Minimum prompts per demographic group (default: "
            f"{DEFAULT_LLM_MIN_GROUP_SIZE}; classifier FairnessAnalyzer uses 30). "
            "Groups below threshold are excluded; metric is nan when fewer than two "
            "eligible groups remain — same semantics as classifier fairness metrics."
        ),
    )
    p.add_argument(
        "--allow-small-samples",
        action="store_true",
        help=(
            "Illustrative-only override: compute metrics even when groups are below "
            "--min-group-size. Results are not production-grade."
        ),
    )
    p.add_argument("--with-ci", action="store_true", help="Compute bootstrap CI for metrics")
    p.add_argument("--ci-level", type=float, default=0.95, help="Confidence level (default 0.95)")
    p.add_argument("--bootstrap-B", type=int, default=200, help="Bootstrap resamples")
    p.add_argument("--random-state", type=int, default=42, help="Bootstrap RNG seed")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional: gate the selected --metric (exit 0 pass / 1 fail / 3 illustrative). "
            "A caveated (illustrative) metric always exits 3, even when the number would "
            "pass the threshold. Requires --metric when set."
        ),
    )
    p.add_argument(
        "--metric",
        default=None,
        help=(
            "LLM-eval metric key to gate (required when --threshold is set). "
            "Without --threshold, a caveat on this metric still exits 3."
        ),
    )
    p.set_defaults(func=cmd_llm_eval)


def add_llm_eval_subparser(sub, *, set_defaults: bool = True) -> argparse.ArgumentParser:
    """Register llm-eval subcommand; exposed for tests."""
    register_llm_eval_parser(sub)
    return sub.choices["llm-eval"]
