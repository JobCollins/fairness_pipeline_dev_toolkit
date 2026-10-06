"""Part 3b addendum — T in {3, 4} (below every shipped fixture) for the min-template question."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import part3b_llm_template_sim as sim

OUT = Path(__file__).resolve().parent / "results"


def main():
    rows = []
    for T in (3, 4, 5):
        for D in (1, 2, 3):
            for rho in sim.RHOS:
                for delta in (0.0, 0.03):
                    rows.extend(sim.run_cell(T, D, rho, delta))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part3b_small_T.csv", index=False)
    df["m"] = df.method.str.split(" ").str[0]
    pd.set_option("display.width", 200)
    print(
        df.pivot_table(
            index=["metric", "m"], columns="T", values="coverage", aggfunc=["mean", "min"]
        ).round(3)
    )
    print(
        df.pivot_table(index=["metric", "m"], columns="T", values="width", aggfunc="mean").round(4)
    )


if __name__ == "__main__":
    main()
