#!/usr/bin/env python3
"""
Generate a development CSV with columns:
  f0, f1, ..., f{d-1}, y, s

- s is a binary sensitive attribute (0/1)
- y is binary and depends on f0 (signal) + a small bias from s
- features are standard normal; f0 carries true signal

Usage:
  python dev_sample_csv.py --n 1000 --d 6 --seed 0 --out dev_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def make_data(n: int, d: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)  # features f0..f{d-1}
    s = (rng.rand(n) > 0.5).astype(int)  # balanced sensitive attribute
    # logistic link: y* depends on f0 + small unfair dependence on s
    logits = 0.9 * X[:, 0] + 0.3 * s + 0.1 * rng.randn(n)
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n) < probs).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(d)}
    df = pd.DataFrame(cols)
    df["y"] = y
    df["s"] = s
    return df, y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of rows")
    ap.add_argument("--d", type=int, default=6, help="number of feature columns f0..f{d-1}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="dev_sample_train.csv", help="output CSV path")
    args = ap.parse_args()

    df, _, _ = make_data(args.n, args.d, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path.resolve()}")
    print(f"Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
