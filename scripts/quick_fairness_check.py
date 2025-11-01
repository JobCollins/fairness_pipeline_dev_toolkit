"""
Quick fairness smoke check.

Exit code:
  0 -> pass (or metric NaN with warning)
  2 -> fail (metric exceeds threshold)

Examples:
  # with defaults: y, yhat, group
  python scripts/quick_fairness_check.py \
    --csv dev_sample.csv \
    --out-md artifacts_demo_report.md \
    --out-json artifacts_demo_report.json

  # explicit column names
  python scripts/quick_fairness_check.py \
    --csv path/to/data.csv \
    --y-true y \
    --y-pred score_bin \
    --sensitive sensitive_attr
"""

from __future__ import annotations

import os
import sys

try:
    import fairness_pipeline_dev_toolkit  # noqa: F401
except ModuleNotFoundError:
    # add repo root (parent of /scripts) to sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


import argparse
import json
import sys
from datetime import datetime

import pandas as pd

from fairness_pipeline_dev_toolkit.measurement import FairnessAnalyzer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with predictions/labels/sensitive columns")
    ap.add_argument("--y-true", default="y_true", help="Ground-truth label column (default: y)")
    ap.add_argument(
        "--y-pred",
        default="y_pred",
        help="Predicted label or score-binarized column (default: yhat)",
    )
    ap.add_argument(
        "--sensitive", default="sensitive", help="Sensitive attribute column (default: group)"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Max allowed demographic parity diff (default: 0.10)",
    )
    ap.add_argument(
        "--min-group-size", type=int, default=30, help="Minimum per-group size (default: 30)"
    )
    ap.add_argument("--out-md", default=None, help="Optional path to write a markdown summary")
    ap.add_argument("--out-json", default=None, help="Optional path to write a JSON summary")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)

    # Validate columns
    for col in (args.y_true, args.y_pred, args.sensitive):
        if col not in df.columns:
            print(
                f"Missing column '{col}' in {args.csv}. Columns are: {list(df.columns)}",
                file=sys.stderr,
            )
            return 2

    # Run a simple fairness check (demographic parity diff)
    fa = FairnessAnalyzer(min_group_size=args.min_group_size, backend="native")
    res = fa.demographic_parity_difference(
        y_pred=df[args.y_pred].to_numpy(),
        sensitive=df[args.sensitive].to_numpy(),
    )
    val = float(res.value) if res.value is not None else float("nan")

    # Decide pass/fail
    status = "pass"
    exit_code = 0
    msg = ""

    if val != val:  # NaN
        msg = (
            "Fairness check WARNING: metric is NaN (likely insufficient data in one or more groups); "
            "skipping gate."
        )
        print(msg, file=sys.stderr)
        status = "warn"
        exit_code = 0
    elif val > args.threshold:
        msg = f"Fairness check FAILED: dpd={val:.6f} > threshold={args.threshold:.6f}"
        print(msg, file=sys.stderr)
        status = "fail"
        exit_code = 2
    else:
        msg = f"Fairness check OK: dpd={val:.6f} <= threshold={args.threshold:.6f}"
        print(msg)
        status = "pass"
        exit_code = 0

    # Optional artifacts
    stamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    summary = {
        "timestamp_utc": stamp,
        "metric": "demographic_parity_difference",
        "value": None if val != val else val,
        "threshold": args.threshold,
        "status": status,
        "columns": {"y_true": args.y_true, "y_pred": args.y_pred, "sensitive": args.sensitive},
        "csv": args.csv,
        "min_group_size": args.min_group_size,
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.out_md:
        md = [
            "# Quick Fairness Check",
            "",
            "- **Metric**: Demographic Parity Difference",
            f"- **Value**: {'NaN' if val != val else f'{val:.6f}'}",
            f"- **Threshold**: {args.threshold}",
            f"- **Status**: **{status.upper()}**",
            "",
            "## Context",
            f"- CSV: `{args.csv}`",
            f"- Columns: y_true=`{args.y_true}`, y_pred=`{args.y_pred}`, sensitive=`{args.sensitive}`",
            f"- Min group size: {args.min_group_size}",
            f"- Timestamp (UTC): {summary['timestamp_utc']}",
            "",
        ]
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
