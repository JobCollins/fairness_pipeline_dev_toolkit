"""Render results/*.csv into results/tables.md (compact tables used in README.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    df = df.copy()
    df.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def t3a(lines):
    d = pd.read_csv(R / "part3a_gap_grid.csv")
    iv = d[d.coverage.notna()].copy()
    iv["m"] = iv.method.str.split(" ").str[0]
    order = ["M0", "M1", "M2a", "M2b", "M4", "M5a", "M5b", "M5c"]
    lines.append(
        "## 3a-1  Coverage of the true gap = 0 (mean of base 0.1 / 0.5; per-cell MC SE <= 0.022)\n"
    )
    z = iv[iv.gap == 0].pivot_table(index=["K", "n"], columns="m", values="coverage")[order]
    lines.append(md(z))
    lines.append("\n## 3a-2  Coverage of the true gap, mean (min) over the 24 (K, n, base) cells\n")
    g = iv.groupby(["m", "gap"]).coverage.agg(["mean", "min"])
    tab = g.apply(lambda r: f"{r['mean']:.3f} ({r['min']:.3f})", axis=1).unstack("gap").loc[order]
    lines.append(md(tab))
    lines.append("\n## 3a-3  Mean interval width\n")
    for K, n in ((3, "100"), (2, "1000"), (5, "1000"), (3, "unequal")):
        w = iv[(iv.K == K) & (iv.n == n) & (iv.base == 0.5)].pivot_table(
            index="m", columns="gap", values="width"
        )
        lines.append(f"\nK={K}, n={n}, base 0.5:\n")
        lines.append(md(w.loc[order], "{:.4f}"))
    lines.append(
        "\n## 3a-4  Share of datasets whose CI excludes its own point estimate (max over cells)\n"
    )
    lines.append(md(iv.groupby("m").excl_point.max().loc[order].to_frame()))
    pt = d[d.reject.notna()].copy()
    pt["stat"] = pt.method.str.extract(r"\((.*) stat\)")[0]
    lines.append(
        "\n## 3a-5  M3 permutation test: rejection rate at alpha=0.05, mean (max) over cells (MC SE ~0.010 at 0.05)\n"
    )
    g = pt.groupby(["stat", "gap"]).reject.agg(["mean", "max"])
    lines.append(md(g.apply(lambda r: f"{r['mean']:.3f} ({r['max']:.3f})", axis=1).unstack("gap")))
    lines.append("\nType-I error by design (gap statistic):\n")
    lines.append(
        md(
            pt[(pt.gap == 0) & (pt.stat == "gap")].pivot_table(
                index="K", columns="n", values="reject", aggfunc="max"
            )
        )
    )
    lines.append("\nPower by design, gap statistic, base 0.5:\n")
    lines.append(
        md(
            pt[(pt.stat == "gap") & (pt.base == 0.5)].pivot_table(
                index=["K", "n"], columns="gap", values="reject"
            )
        )
    )
    e = pd.read_csv(R / "part3a_equivalence.csv")
    lines.append(
        "\n## 3a-6  Equivalence: P(upper bound < 0.10), base 0.5 (must be <= 0.05 when true gap >= 0.10)\n"
    )
    lines.append(
        md(
            e.pivot_table(
                index=["K", "n", "method"], columns="true_gap", values="P(declare gap<0.10)"
            )
        )
    )
    lines.append("\n## 3a-7  Runtime, ms per simulated dataset (B=1000, mean over cells)\n")
    lines.append(md(d.groupby("method").ms_per_dataset.mean().to_frame(), "{:.2f}"))


def t3b(lines):
    d = pd.concat(
        [pd.read_csv(R / "part3b_llm_template_sim.csv"), pd.read_csv(R / "part3b_small_T.csv")]
    ).drop_duplicates(subset=["T", "D", "rho", "delta", "metric", "method"])
    d["m"] = d.method.str.split(" ").str[0]
    for metric in ("divergence", "contrast"):
        x = d[(d.metric == metric) & d["T"].isin([5, 10, 20, 50]) & d.delta.isin([0.0, 0.03])]
        lines.append(
            f"\n## 3b-{metric}  Coverage, mean over rho in {{0, .3, .7}} (per-cell MC SE <= 0.022)\n"
        )
        tab = x.pivot_table(index=["D", "delta", "m"], columns="T", values="coverage")
        lines.append(md(tab))
    x = d[d.delta.isin([0.0, 0.03])]
    lines.append("\n## 3b-minT  Coverage mean (min) over D, rho, delta by T\n")
    g = x.groupby(["metric", "m", "T"]).coverage.agg(["mean", "min"])
    lines.append(md(g.apply(lambda r: f"{r['mean']:.3f} ({r['min']:.3f})", axis=1).unstack("T")))
    lines.append("\n## 3b-width  Mean width, D=2, delta=0, rho=0.3\n")
    w = d[(d.D == 2) & (d.delta == 0) & (d.rho == 0.3)].pivot_table(
        index=["metric", "m"], columns="T", values="width"
    )
    lines.append(md(w, "{:.4f}"))
    lines.append("\n## 3b-rho  Coverage by rho, D=2, delta=0\n")
    r = d[(d.D == 2) & (d.delta == 0) & d["T"].isin([5, 10, 50])].pivot_table(
        index=["metric", "m"], columns=["rho", "T"], values="coverage"
    )
    lines.append(md(r))
    lines.append(
        "\n## 3b-excl  Share of datasets whose CI excludes the reported value (max over cells)\n"
    )
    lines.append(
        md(d.pivot_table(index=["metric", "m"], columns="T", values="excl_point", aggfunc="max"))
    )
    x = d[(d.metric == "contrast") & (d.delta == 0)]
    lines.append(
        "\n## 3b-null  Contrast with true value 0: share of CIs excluding 0, mean (max) over D, rho\n"
    )
    g = x.groupby(["m", "T"]).excl0.agg(["mean", "max"])
    lines.append(md(g.apply(lambda r: f"{r['mean']:.3f} ({r['max']:.3f})", axis=1).unstack("T")))


def t3c(lines):
    d = pd.read_csv(R / "part3c_bca_nan_policy.csv")
    lines.append(
        "\n## 3c  BCa NaN policies: unconditional coverage / undefined share (S=500, B=1000)\n"
    )
    d["cell"] = d.apply(lambda r: f"{r.coverage_uncond:.2f} / {r.undefined:.2f}", axis=1)
    for g in (0.0, 0.2):
        lines.append(
            f"\nTrue gap {g} (majority groups n=300; NaN share per replicate in last column):\n"
        )
        tab = d[d.gap == g].pivot_table(
            index=["K", "n_minor"], columns="policy", values="cell", aggfunc="first"
        )
        tab["NaN share"] = (
            d[d.gap == g].groupby(["K", "n_minor"]).mean_nan_share.first().map("{:.4f}".format)
        )
        lines.append(md(tab))


def main():
    lines = [
        "# Wave 3a CI investigation — generated tables\n",
        "Generated by make_tables.py from results/*.csv.\n",
    ]
    t3a(lines)
    t3b(lines)
    t3c(lines)
    (R / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", R / "tables.md")


if __name__ == "__main__":
    main()
