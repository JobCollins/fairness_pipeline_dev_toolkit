"""
`fairpipe` CLI

Subcommands:
- version
- validate : run a quick fairness validation on a CSV and print/save a Markdown report

Examples:
    python -m fairness_pipeline_dev_toolkit.cli.main version

    python -m fairness_pipeline_dev_toolkit.cli.main validate \
        --csv data/holdout.csv \
        --y-true y_true --y-pred y_pred \
        --sensitive race --sensitive sex \
        --min-group-size 30 \
        --out report.md
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import pandas as pd

from fairness_pipeline_dev_toolkit.integration.reporting import to_markdown_report
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer


def cmd_version(args: argparse.Namespace) -> int:
    from fairness_pipeline_dev_toolkit import __version__

    print(__version__)
    return 0


def _parse_sensitive(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        raise SystemExit("No --sensitive columns provided.")
    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"Sensitive column not found: {c}")
    return df[cols]


def _normalize_sensitive_arg(sens_arg) -> list[str]:
    """
    Accepts either a string "col1,col2" or a list ["col1", "col2"] (possibly with commas),
    and returns a clean list of column names.
    """
    if isinstance(sens_arg, (list, tuple)):
        items = []
        for item in sens_arg:
            items.extend([p.strip() for p in str(item).split(",") if p.strip()])
        return items
    else:
        return [p.strip() for p in str(sens_arg).split(",") if p.strip()]


def cmd_validate(args: argparse.Namespace) -> int:
    # 1) Load
    df = pd.read_csv(args.csv)

    # 2) Basic arg checks
    if not args.y_true:
        raise SystemExit("--y-true is required")
    if (args.y_pred is None) == (args.score is None):
        raise SystemExit("Provide exactly one of --y-pred OR --score")
    if not args.sensitive:
        raise SystemExit("--sensitive is required (single col or comma-separated)")

    # 3) Resolve sensitive columns (supports comma-separated)
    sens_cols = _normalize_sensitive_arg(args.sensitive)
    missing = [c for c in sens_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"--sensitive columns not found: {missing}")

    # 4) Build sensitive vector and (optional) attrs_df
    if len(sens_cols) == 1:
        sensitive_col = df[sens_cols[0]].to_numpy().ravel()
        attrs_df = None  # or df[[sens_cols[0]]] if you need a DF elsewhere
    else:
        # intersectional labels like "race_gender_age"
        sensitive_col = df[sens_cols].astype(str).agg("_".join, axis=1).to_numpy().ravel()
        attrs_df = df[sens_cols].copy()

    # 5) Targets / predictions
    if args.y_true not in df.columns:
        raise SystemExit(f"--y-true column not found: {args.y_true}")
    y_true = df[args.y_true].to_numpy().ravel()

    y_pred = None
    scores = None
    if args.y_pred:
        if args.y_pred not in df.columns:
            raise SystemExit(f"--y-pred column not found: {args.y_pred}")
        y_pred = df[args.y_pred].to_numpy().ravel()

    if args.score:
        if args.score not in df.columns:
            raise SystemExit(f"--score column not found: {args.score}")
        scores = df[args.score].to_numpy().ravel()

    # 6) Optional: drop rows with any missing among required columns
    required = (
        [args.y_true]
        + sens_cols
        + ([args.y_pred] if args.y_pred else [])
        + ([args.score] if args.score else [])
    )
    mask = df[required].notna().all(axis=1)
    if not mask.all():
        # down-select arrays consistently
        y_true = y_true[mask.values]
        sensitive_col = sensitive_col[mask.values]
        if y_pred is not None:
            y_pred = y_pred[mask.values]
        if scores is not None:
            scores = scores[mask.values]
        if attrs_df is not None:
            attrs_df = attrs_df.loc[mask.values]

    # 7) Length alignment check (defensive)
    n = len(y_true)
    if (
        len(sensitive_col) != n
        or (y_pred is not None and len(y_pred) != n)
        or (scores is not None and len(scores) != n)
    ):
        raise SystemExit(
            "Mismatched lengths among inputs after filtering — check your CSV and column choices."
        )

    # 8) Analyzer
    fa = FairnessAnalyzer(min_group_size=args.min_group_size, backend=args.backend)

    results = {}

    # 9) Classification metrics (if y_pred present)
    if y_pred is not None:
        results["demographic_parity_difference"] = fa.demographic_parity_difference(
            y_pred=y_pred, sensitive=sensitive_col
        )
        # Only include EO if your implementation is ready/needed
        results["equalized_odds_difference"] = fa.equalized_odds_difference(
            y_true=y_true, y_pred=y_pred, sensitive=sensitive_col
        )

    # 10) Regression metric (scores + y_true)
    if scores is not None:
        results["mae_parity_difference"] = fa.mae_parity_difference(
            y_true=y_true, y_pred=scores, sensitive=sensitive_col
        )

    # 11) Markdown report
    md = to_markdown_report(results, title="Fairness Validation Report (CLI)")
    print(md)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)

    return 0


def cmd_sample_check(args):
    """Placeholder for sample check (non-blocking pre-commit)."""
    print("Running fairness sample check (placeholder)...")
    # lightweight smoke logic here, e.g. check for existence of dev_sample.csv
    import os

    if not os.path.exists("dev_sample.csv"):
        print("⚠️  dev_sample.csv not found — skipping check.")
        return 0
    print("✅ Sample data exists. Pre-commit check passed.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="fairpipe", description="Fairness Toolkit CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ver = sub.add_parser("version", help="Print package version")
    p_ver.set_defaults(func=cmd_version)

    p_val = sub.add_parser("validate", help="Run a quick fairness validation from a CSV")
    p_val.add_argument("--csv", required=True, help="Path to CSV with predictions and attributes")
    p_val.add_argument("--y-true", required=True, help="Column with ground-truth labels/targets")
    p_val.add_argument("--y-pred", help="Column with predicted labels (for classification)")
    p_val.add_argument(
        "--score", help="Column with predicted scores (for regression or thresholds)"
    )
    p_val.add_argument(
        "--sensitive", nargs="+", required=True, help="Sensitive attribute column(s)"
    )
    p_val.add_argument("--min-group-size", type=int, default=30, help="Minimum group size")
    p_val.add_argument(
        "--backend",
        choices=["auto", "native", "fairlearn", "aequitas"],
        default="auto",
        help="Metric backend selection",
    )
    p_val.add_argument("--out", help="Write Markdown report to this file")
    p_val.set_defaults(func=cmd_validate)

    # NEW: add sample-check
    sample = sub.add_parser("sample-check")
    sample.set_defaults(func=cmd_sample_check)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
