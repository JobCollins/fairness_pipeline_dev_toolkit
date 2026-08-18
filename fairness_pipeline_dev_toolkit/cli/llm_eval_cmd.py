from __future__ import annotations

import argparse

from fairness_pipeline_dev_toolkit.llm_evals.config import load_llm_eval_config
from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE
from fairness_pipeline_dev_toolkit.llm_evals.runner import (
    results_to_markdown,
    run_llm_eval,
    write_transcripts,
)


def cmd_llm_eval(args: argparse.Namespace) -> int:
    config = load_llm_eval_config(path=args.config)
    from fairness_pipeline_dev_toolkit.llm_evals.runner import build_client

    client = build_client(config) if config.provider == "local" and not args.dry_run else None
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

    if args.dry_run:
        md = result.dry_run.to_markdown() if result.dry_run else ""
    else:
        md = results_to_markdown(result, title="LLM Fairness Evaluation Report")

    print(md)

    report_path = args.report_md or args.out
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)

    if args.transcripts_out and result.transcripts:
        write_transcripts(args.transcripts_out, result.transcripts)

    return 0


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
    p.set_defaults(func=cmd_llm_eval)


def add_llm_eval_subparser(sub, *, set_defaults: bool = True) -> argparse.ArgumentParser:
    """Register llm-eval subcommand; exposed for tests."""
    register_llm_eval_parser(sub)
    return sub.choices["llm-eval"]
