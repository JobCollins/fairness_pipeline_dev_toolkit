"""Does the COMPAS notebook's 'both CIs sit entirely above zero' DPD claim survive M2a / M3?"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from runtime_case_study_sizes import m2a_ac, permutation_gap

from fairness_pipeline_dev_toolkit.metrics.core import FairnessAnalyzer

ROOT = Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "case_studies" / "compas_bw.csv")
y, s = d["y_pred"].to_numpy(), d["race"].to_numpy()
codes, uniq = pd.factorize(s)
r = FairnessAnalyzer(min_group_size=30, backend="native").demographic_parity_difference(
    y, s, ci_samples=1000, with_effect_size=False
)
out = {
    "groups": {str(g): int((s == g).sum()) for g in uniq},
    "dpd": r.value,
    "current_percentile_ci": list(r.ci),
    "M2a_simultaneous": list(m2a_ac(y, codes, len(uniq))),
    "M3_permutation_p_B=10000": permutation_gap(
        y, codes, len(uniq), 10_000, np.random.default_rng(0)
    ),
}
(Path(__file__).parent / "results" / "compas_claim_check.json").write_text(
    json.dumps(out, indent=1)
)
print(json.dumps(out, indent=1))
