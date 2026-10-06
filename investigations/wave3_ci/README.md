# Wave 3a — Bootstrap confidence intervals: investigation

Investigation only. Nothing under `fairness_pipeline_dev_toolkit/` or `fairpipe/` was changed.
Backlog items: BL-014 (#24), BL-016 (#26), BL-031 (#41).

AI-generated (Cursor agent); numbers are reproducible from the scripts below.

## How to run

From the repo root, with the project venv:

```bash
cd investigations/wave3_ci
export PYTHONPATH="$(cd ../.. && pwd):$PWD"
../../.venv/bin/python part1_llm_cluster_structure.py
../../.venv/bin/python part2_repro.py
../../.venv/bin/python check_vectorised_vs_package.py
../../.venv/bin/python part3a_gap_grid.py        # ~2 min
../../.venv/bin/python part3a_equivalence.py
../../.venv/bin/python part3b_llm_template_sim.py
../../.venv/bin/python part3b_small_T.py
../../.venv/bin/python part3b_fixtures.py
../../.venv/bin/python part3c_bca_nan_policy.py
../../.venv/bin/python runtime_case_study_sizes.py
../../.venv/bin/python compas_claim_check.py
../../.venv/bin/python b_stability.py
../../.venv/bin/python make_tables.py            # -> results/tables.md
```

No network and no live LLM calls: synthetic data, offline fake clients, and the shipped
fixtures in `llm_evals/fixtures/` only. All seeds are fixed.

| File | Purpose |
|---|---|
| `gapsim.py` | Vectorised method library (M0–M5, cluster methods). Cross-checked against the package in `check_vectorised_vs_package.py` (BCa identical to 1e-15; M0 coverage matches the real analyzer within MC SE). |
| `part1_*` | LLM cluster structure of every shipped fixture. |
| `part2_repro.py` | Deterministic reproductions of BL-016 / BL-014 / BL-031. |
| `part3a_*` | Max−min gap grid (S=500, B=1000, 120 cells × 10 methods) and equivalence decisions. |
| `part3b_*` | LLM template dependence (S=500, B=1000) and the shipped fixtures (B=10000). |
| `part3c_*` | BCa NaN policies. |
| `runtime_*`, `compas_claim_check.py`, `b_stability.py` | Runtime, published-claim check, B sensitivity. |
| `results/tables.md` | All simulation tables (generated). |

MC SE for a coverage estimate is `sqrt(p(1-p)/S)`: ≤0.022 per cell at S=500, 0.010 at p=0.95.

## Method labels

Analyzer / rate-disparity gaps (statistic = max over groups − min over groups):

- **M0** today's analyzer: unstratified index bootstrap, percentile.
- **M1** stratified (within-group) bootstrap, percentile (= today's `bootstrap_rate_disparity`).
- **M2a** simultaneous-interval inversion, Bonferroni Agresti–Caffo over all pairs (analytic).
- **M2b** simultaneous-interval inversion, stratified bootstrap max-|t| over all pairs.
  For both: `L = max_pairs max(0, l_ij, -u_ij)`, `U = max_pairs max(|l_ij|, |u_ij|)`;
  coverage ≥ 1−α by construction because the pairwise family is simultaneous.
- **M3** permutation test of "all group rates equal" (gap statistic or chi-square).
- **M4** m-out-of-n bootstrap, common fraction `f = n_min^-0.3`, root interval.
- **M5a/b/c** BCa as-is / NaN-dropped / on stratified replicates.

LLM divergence / contrast:

- **C0** today: iid bootstrap of pooled pair values, `np.mean` (not the reported statistic).
- **C1** template-cluster bootstrap recomputing the exact reported statistic, percentile.
- **C2a** template-cluster bootstrap, simultaneous max-|t| inversion over dimensions.
- **C2b** Bonferroni-t inversion over dimensions on template-level means (df = T−1);
  contrast uses template-paired differences against the control arm.
- **C2c** m-out-of-n template-cluster bootstrap.

## Findings, short

1. **BL-014 is structural.** The percentile bootstrap of a max−min statistic has a lower bound
   ≥ 0 and covers 0 only via lattice ties. Coverage of a true gap of 0 is 0.000 for K ≥ 3 at
   every n tested, and still 0.56 mean (min 0.000) at a true gap of 0.02. Stratifying alone
   (M1) does not fix it; BCa (M5) does not fix it. M2a covers ≥ 0.938 at gap 0 in every cell
   (mean 0.960) and ≥ 0.922 everywhere.
2. **BL-016 is a statistic mismatch plus a unit mismatch.** Today's divergence/contrast CI is a
   CI for the pooled mean of pair values resampled as iid; the reported value is a max of
   dimension means. Both CIs exclude their own point in the reproduction; in simulation C0
   excludes the reported value in up to 100% of datasets and covers 0.000 at T=50 when
   dimensions differ. Resampling templates and recomputing the exact statistic (C1) fixes
   the mismatch but still undercovers (0.70–0.85 at small T or equal dimensions). C2b is the
   only method ≥ 0.93 in every cell, down to T=3.
3. **BL-031 is silent NaN, not a shifted interval.** NaN replicates make `np.percentile`
   return NaN (or raise when the jackknife acceleration is NaN). The `n < 5 / non-finite`
   guard checks the index array and never fires. With the default `min_group_size=30` the
   per-replicate miss probability is ~e^-30, so it only bites when users lower the floor.
   No NaN policy produces a calibrated BCa interval for tiny groups; stratifying removes the
   NaN source but BCa still undercovers at the boundary.

All simulation tables are in `results/tables.md`; raw per-cell results are in `results/*.csv`.

## Recommendations

### BL-014 — max−min gaps (DPD, EOD, MAE gap, refusal/toxicity/stereotype disparity)

- Default interval: **M2 simultaneous-interval inversion**. M2a (Bonferroni Agresti–Caffo) for
  binary rates: analytic, no RNG, no B. For continuous per-group statistics (MAE gap,
  toxicity scores) Agresti–Caffo does not apply; use M2b with stratified resampling (min
  coverage 0.856 at gap 0 with a 30-row skewed group; mean 0.939) or a Bonferroni-Welch
  variant (not simulated). EOD needs the family over all TPR and FPR pairs (not simulated).
- Width cost vs today: +0% at K=2 n=1000, +19% at K=3 n=100, +52% at K=5 n=1000. That is the
  price of a guarantee; today's narrower interval is narrow because it is wrong at the boundary.
- Offer **M3 permutation test** (gap statistic) as a separate `p_value` field for
  "all group rates equal". Type-I error 0.022–0.054 across the grid. Within-template
  permutation is required for paired LLM designs (not simulated).
- Offer the **equivalence decision** "upper bound < δ". M2 upper bound: P(declare | true gap = δ)
  ≤ 0.016. Today's percentile upper bound was also valid in the cells tested (≤ 0.020), but has
  no guarantee. M2 has less equivalence power at K=5 (0.59 vs 0.89 at true gap 0).
- Wording: never "CI excludes zero ⇒ significant". Suggested:
  - `L > 0`: "The largest between-group gap is at least L (95% simultaneous interval [L, U])."
  - `L = 0`: "The data are consistent with no gap; the gap could be as large as U."
  - `U < δ`: "The gap is below δ with 95% confidence."
  - Significance only from the permutation p-value.

### BL-016 — LLM divergence / contrast

- Resampling unit: **template** (`replicate_id`). A draw takes all responses for a template
  across every dimension, group and the control arm together.
- Statistic: the exact reported value (max over gated dimensions of dimension means; contrast
  = that minus the control mean from the same templates).
- Interval: **C2b**. Coverage mean 0.967–0.973 and min 0.930 over every cell, T = 3–50.
  Contrast false "excludes 0" rate 0.03–0.04. Width at D=2: 0.155 (T=3), 0.093 (T=4),
  0.072 (T=5), 0.040 (T=10), 0.016 (T=50).
- Minimum templates: CI undefined for **T < 5** (width at T=3–4 is 2–4× T=10, and the normal
  approximation of bounded template means is untested there); report with a "few templates"
  caveat for **5 ≤ T < 10**. T=2 is degenerate (t with 1 df). The recorded single-template
  fixture (T=1) must return an undefined CI.

### BL-031 — BCa NaN policy

- Stratify analyzer resampling by group (and by group × `y_true` for EOD). This removes NaN
  replicates by construction; today EOD silently drops a group's TPR in 13.6% of replicates
  in the tested design because `nanmax` hides it.
- Policy: **refuse** — undefined CI with a reason when any group is below the floor or the
  jackknife is non-finite. Never compute z0 or quantiles over NaN. Fallback-to-percentile
  inherits BL-014; drop-NaN conditions on "all groups present" and is undefined for
  minority size ≤ 5 at a 0.99 floor.
- Consider deprecating `ci_method="bca"` for max−min metrics: its coverage of a true gap of 0
  is 0.000 for K ≥ 3.

### Default B

| Path | Today | Recommended |
|---|---|---|
| Analyzer M2a / LLM C2b | 1000 / 200 | none (analytic) |
| Analyzer M2b, M3 permutation | — | 2000 |
| Any remaining percentile bootstrap (e.g. `bootstrap_ci`) | 200 (LLM) | ≥ 2000 |

At B=200 the seed-to-seed SD of each hiring-fixture endpoint is 0.0011 (4.7% of width);
B=2000 gives 0.00036 (1.5%) (`results/b_stability.json`).
