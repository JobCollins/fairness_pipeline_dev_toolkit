"""Part 3a (equivalence) — how often "upper bound < threshold" is declared.

Decision: declare "gap < delta" when the interval's upper bound is below delta.
For a valid level-alpha equivalence decision, P(declare) must be <= alpha whenever
the true gap >= delta. Compared: M2a / M2b upper bounds (two-sided 95% family, so
the one-sided error is <= 2.5% by construction) and the current M0 percentile upper
bound (the pattern docs/integration_guide.md shows: ``assert result.ci[1] < 0.10``).

delta = 0.10; base rate 0.5; S=500, B=1000.
"""

from __future__ import annotations

from pathlib import Path

import gapsim as G
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
S, B, DELTA = 500, 1000, 0.10
DESIGNS = {
    (2, "100"): [100, 100],
    (2, "1000"): [1000, 1000],
    (3, "100"): [100] * 3,
    (3, "1000"): [1000] * 3,
    (3, "unequal"): [30, 300, 3000],
    (5, "1000"): [1000] * 5,
}
GAPS = (0.0, 0.05, 0.08, 0.10, 0.12)


def main():
    rows = []
    for (K, label), sizes in DESIGNS.items():
        n = np.array(sizes)
        for g in GAPS:
            p = np.full(K, 0.5)
            p[0] += g
            ss = np.random.SeedSequence([K, len(label), int(g * 100), 99])
            rd, r0, r1 = (np.random.default_rng(x) for x in ss.spawn(3))
            X = rd.binomial(n[None, :], p[None, :], size=(S, K))
            g0, _ = G.m0_replicates(r0, X, n, B)
            _, ub0 = G.percentile(g0)
            _, uba = G.m2_bonferroni_ac(X, n)
            _, ubb = G.m2_maxt(X, n, G.m1_rates(r1, X, n, B))
            for meth, ub in (("M0 pct upper", ub0), ("M2a upper", uba), ("M2b upper", ubb)):
                r = float(np.mean(ub < DELTA))
                rows.append(
                    {
                        "K": K,
                        "n": label,
                        "true_gap": g,
                        "method": meth,
                        "P(declare gap<0.10)": r,
                        "mcse": np.sqrt(r * (1 - r) / S),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part3a_equivalence.csv", index=False)
    print(
        df.pivot_table(
            index=["K", "n", "method"], columns="true_gap", values="P(declare gap<0.10)"
        ).round(3)
    )


if __name__ == "__main__":
    main()
