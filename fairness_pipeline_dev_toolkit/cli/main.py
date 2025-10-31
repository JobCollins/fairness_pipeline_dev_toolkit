"""
`fairpipe` CLI

Subcommands:
- version
- validate : run a quick fairness validation on a CSV and print/save a Markdown report
- sample-check : lightweight pre-commit check on sample data
- pipeline-run : (NEW Pipeline Module Phase 0 demo)

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
import pathlib
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.integration.reporting import to_markdown_report
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer
from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
    run_detectors,
)
from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci
from fairness_pipeline_dev_toolkit.stats.effect_size import risk_ratio

# import yaml


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


def _bootstrap_metric_ci(n, stat_fn, B=1000, level=0.95):
    """
    Generic bootstrap over row indices. stat_fn(idx: np.ndarray[int]) -> float
    """
    idx = np.arange(n, dtype=int)

    def stat_from_idx(sample_idx):
        return float(stat_fn(sample_idx))

    return bootstrap_ci(idx, stat_from_idx, B=B, level=level, method="percentile")


def _dp_risk_ratio(
    y_pred: np.ndarray, sensitive: np.ndarray, min_group_size: int
) -> Optional[float]:
    # selection rates per group
    s = pd.Series(sensitive)
    yp = np.asarray(y_pred)
    counts = s.value_counts()
    valid_mask = s.map(counts).to_numpy() >= min_group_size
    if valid_mask.sum() == 0:
        return None
    s = s[valid_mask].to_numpy()
    yp = yp[valid_mask]
    groups = np.unique(s)
    if len(groups) < 2:
        return None
    rates = []
    for g in groups:
        m = s == g
        rates.append(yp[m].mean())
    hi = float(np.max(rates))
    lo = float(np.min(rates))
    if hi == 0.0 or lo == 0.0:
        return None
    return risk_ratio(hi, lo)


def _eod_ratio(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, min_group_size: int
) -> Optional[float]:
    """
    Compute a single 'effect size' proxy for EOD: the max of (TPR ratio, FPR ratio)
    across groups (largest ratio across all group pairs). Returns None if undefined.
    """
    s = pd.Series(sensitive)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    counts = s.value_counts()
    valid_mask = s.map(counts).to_numpy() >= min_group_size
    if valid_mask.sum() == 0:
        return None
    s = s[valid_mask].to_numpy()
    yt = yt[valid_mask]
    yp = yp[valid_mask]
    groups = np.unique(s)
    if len(groups) < 2:
        return None

    # per-group TPR/FPR
    tprs, fprs = [], []
    for g in groups:
        m = s == g
        yt_g, yp_g = yt[m], yp[m]
        pos = yt_g == 1
        neg = yt_g == 0
        tpr = float(yp_g[pos].mean()) if pos.any() else np.nan
        fpr = float(yp_g[neg].mean()) if neg.any() else np.nan
        tprs.append(tpr)
        fprs.append(fpr)

    def _max_ratio(arr):
        vals = [v for v in arr if not np.isnan(v) and v > 0]
        if len(vals) < 2:
            return None
        hi, lo = max(vals), min(vals)
        if lo == 0.0:
            return None
        return hi / lo

    candidates = [r for r in (_max_ratio(tprs), _max_ratio(fprs)) if r is not None]
    return max(candidates) if candidates else None


def _cohens_d_between_groups(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, min_group_size: int
) -> Optional[float]:
    """
    Cohen's d between residuals of groups with max vs min MAE.
    """
    s = pd.Series(sensitive)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    counts = s.value_counts()
    valid_mask = s.map(counts).to_numpy() >= min_group_size
    if valid_mask.sum() == 0:
        return None
    s = s[valid_mask].to_numpy()
    yt = yt[valid_mask]
    yp = yp[valid_mask]
    groups = np.unique(s)
    if len(groups) < 2:
        return None

    maes = {}
    residuals_by_group = {}
    res = np.abs(yt - yp)
    for g in groups:
        m = s == g
        maes[str(g)] = float(res[m].mean())
        residuals_by_group[str(g)] = res[m].astype(float)

    if len(maes) < 2:
        return None
    g_max = max(maes, key=maes.get)
    g_min = min(maes, key=maes.get)
    a, b = residuals_by_group[g_max], residuals_by_group[g_min]
    # pooled SD
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if sp == 0:
        return None
    return float((a.mean() - b.mean()) / sp)


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

    # Parse sensitive (support single or multiple)
    if isinstance(args.sensitive, list):
        sens_cols = args.sensitive
    else:
        sens_cols = [c.strip() for c in str(args.sensitive).split(",") if c.strip()]
    if len(sens_cols) == 1:
        sensitive_col = df[sens_cols[0]].to_numpy().ravel()
        attrs_df = df[[sens_cols[0]]]
    else:
        attrs_df = df[sens_cols]
        sensitive_col = attrs_df.apply(lambda r: "_".join(map(str, r.values)), axis=1).to_numpy()

    y_true = df[args.y_true].to_numpy().ravel() if args.y_true else None
    y_pred = df[args.y_pred].to_numpy().ravel() if args.y_pred else None
    scores = df[args.score].to_numpy().ravel() if args.score else None

    fa = FairnessAnalyzer(min_group_size=args.min_group_size, backend=args.backend)
    n = len(df)
    results = {}

    # 1) DPD (classification) if y_pred present
    if y_pred is not None:
        r = fa.demographic_parity_difference(y_pred=y_pred, sensitive=sensitive_col)
        # CI (bootstrap over indices)
        if args.with_ci:

            def stat_fn(idx):
                idx = np.asarray(idx, dtype=int)
                return fa.demographic_parity_difference(
                    y_pred=y_pred[idx], sensitive=sensitive_col[idx]
                ).value

            r.ci = _bootstrap_metric_ci(n, stat_fn, B=args.bootstrap_B, level=args.ci_level)

        # Effect size: risk ratio of selection rates
        if args.with_effects:
            r.effect_size = _dp_risk_ratio(y_pred, sensitive_col, args.min_group_size)

        results["demographic_parity_difference"] = r

    # 2) EOD (classification) if y_pred and y_true
    if (y_pred is not None) and (y_true is not None):
        r = fa.equalized_odds_difference(
            y_true=y_true, y_pred=y_pred, sensitive=sensitive_col, attrs_df=attrs_df
        )
        if args.with_ci:

            def stat_fn(idx):
                idx = np.asarray(idx, dtype=int)
                return fa.equalized_odds_difference(
                    y_true=y_true[idx],
                    y_pred=y_pred[idx],
                    sensitive=sensitive_col[idx],
                    attrs_df=attrs_df.iloc[idx] if attrs_df is not None else None,
                ).value

            r.ci = _bootstrap_metric_ci(n, stat_fn, B=args.bootstrap_B, level=args.ci_level)

        if args.with_effects:
            r.effect_size = _eod_ratio(y_true, y_pred, sensitive_col, args.min_group_size)

        results["equalized_odds_difference"] = r

    # 3) MAE parity (regression-ish) if scores + y_true
    if (scores is not None) and (y_true is not None):
        r = fa.mae_parity_difference(
            y_true=y_true, y_pred=scores, sensitive=sensitive_col, attrs_df=attrs_df
        )
        if args.with_ci:

            def stat_fn(idx):
                idx = np.asarray(idx, dtype=int)
                return fa.mae_parity_difference(
                    y_true=y_true[idx],
                    y_pred=scores[idx],
                    sensitive=sensitive_col[idx],
                    attrs_df=attrs_df.iloc[idx] if attrs_df is not None else None,
                ).value

            r.ci = _bootstrap_metric_ci(n, stat_fn, B=args.bootstrap_B, level=args.ci_level)

        if args.with_effects:
            r.effect_size = _cohens_d_between_groups(
                y_true, scores, sensitive_col, args.min_group_size
            )

        results["mae_parity_difference"] = r

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


def _write_artifact(path: Optional[str], content: str, mode: str = "w") -> None:
    if not path:
        return
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode, encoding="utf-8") as f:
        f.write(content)


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    """
    Run the Pipeline Module:
      1) load config (YAML)
      2) load CSV data
      3) run detectors (optional)
      4) build sklearn pipeline from config
      5) apply pipeline to data
      6) write transformed CSV and optional JSON/Markdown reports
    """
    # 1) Load config
    cfg = load_config(args.config)

    # 2) Load data
    df = pd.read_csv(args.csv)

    # 3) Run detectors (optional)
    detector_report = None
    if not args.no_detectors:
        detector_report = run_detectors(
            df=df,
            cfg=cfg,
            # If your detectors need explicit sensitive column names, pass via config.
        )
        # Pretty print a short summary
        print("== Detector Summary ==")
        for key, val in detector_report.items():
            print(f"- {key}: {val if not isinstance(val, dict) else '[dict]'}")

        # Optionally write a JSON artifact of the full detector output
        if args.detector_json:
            _write_artifact(args.detector_json, str(detector_report))

    # 4) Build pipeline
    pipe = build_pipeline(cfg)

    # 5) Apply pipeline
    Xt, _ = apply_pipeline(pipe, df)

    # 6) Persist outputs
    if args.out_csv:
        _write_artifact(args.out_csv, Xt.to_csv(index=False))

    # Optional Markdown summary of what ran (lightweight)
    if args.report_md:
        lines = ["# Pipeline Run Report", ""]
        lines.append(f"- **Config**: `{args.config}`")
        lines.append(f"- **Input CSV**: `{args.csv}`")
        if args.out_csv:
            lines.append(f"- **Output CSV**: `{args.out_csv}`")
        if detector_report is not None:
            lines.append("")
            lines.append("## Detector Findings (summary)")
            for k, v in detector_report.items():
                if isinstance(v, dict):
                    lines.append(f"- **{k}**: {len(v)} entries")
                else:
                    lines.append(f"- **{k}**: {v}")
        _write_artifact(args.report_md, "\n".join(lines))

    print("Pipeline completed.")
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
    # add to `validate` subparser
    p_val.add_argument("--with-ci", action="store_true", help="Compute bootstrap CI for metrics")
    p_val.add_argument(
        "--ci-level", type=float, default=0.95, help="Confidence level (default 0.95)"
    )
    p_val.add_argument(
        "--bootstrap-B", type=int, default=1000, help="Bootstrap resamples (default 1000)"
    )
    p_val.add_argument("--with-effects", action="store_true", help="Compute effect sizes")

    # NEW: add sample-check
    sample = sub.add_parser("sample-check")
    sample.set_defaults(func=cmd_sample_check)

    # Pipeline
    p_pipe = sub.add_parser("pipeline", help="Run detectors + apply configured pipeline on a CSV")
    p_pipe.add_argument("--config", required=True, help="Path to pipeline.config.yml")
    p_pipe.add_argument("--csv", required=True, help="Input CSV")
    p_pipe.add_argument("--out-csv", help="Write transformed CSV here")
    p_pipe.add_argument("--detector-json", help="Write detector findings JSON here")
    p_pipe.add_argument("--report-md", help="Write a brief Markdown run report here")
    p_pipe.add_argument("--no-detectors", action="store_true", help="Skip detector stage")
    p_pipe.set_defaults(func=cmd_pipeline_run)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
