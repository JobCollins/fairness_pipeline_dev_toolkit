"""
Quick fairness smoke check.

Intended for fast hooks: small sample, one or two metrics, low overhead.
Exit code 0 → pass; non-zero → fail.

Example:
    python scripts/quick_fairness_check.py --csv data/sample.csv --y-true y --y-pred yhat --sensitive group
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--y-true", required=True)
    ap.add_argument("--y-pred", required=True)
    ap.add_argument("--sensitive", required=True)
    ap.add_argument(
        "--threshold", type=float, default=0.10, help="Max allowed demographic parity difference"
    )
    ap.add_argument("--min-group-size", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fa = FairnessAnalyzer(min_group_size=args.min_group_size, backend="auto")
    res = fa.demographic_parity_difference(
        y_pred=df[args.y_pred].to_numpy(),
        sensitive=df[args.sensitive].to_numpy(),
    )
    val = res.value
    if val != val:  # NaN
        print(
            "Fairness check WARNING: metric is NaN (insufficient data); skipping gate.",
            file=sys.stderr,
        )
        return 0
    if val > args.threshold:
        print(
            f"Fairness check FAILED: dpd={val:.6f} > threshold={args.threshold:.6f}",
            file=sys.stderr,
        )
        return 2
    print(f"Fairness check OK: dpd={val:.6f} <= threshold={args.threshold:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
