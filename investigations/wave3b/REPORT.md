# Wave 3b — BBQ fixture validity

**Date:** 2026-10-06  
**Branch:** `wave3b-investigation`  
**HEAD baseline:** `ce6eb7b` (origin/mirror `main`)  
**Scope:** Does the shipped BBQ stereotype probe measure what it claims?
Supersedes the BBQ half of BL-009 for the *construct* question; overlaps
BL-021 / JobCollins#31. Not about CI calibration (that is settled).

Reproduce:

```bash
source .venv/bin/activate
PYTHONPATH=. python investigations/wave3b/probe_bbq_validity.py
# writes investigations/wave3b/results/bbq_validity.json
```

---

## Verdict (recommended outcome)

**Outcome 2 — keep the fixture, but stop calling it a BBQ evaluation.**

The probe is **not** a valid implementation of Parrish et al. (2022) BBQ.
It is a schema-shaped toy with artificial group tags. The recorded cache is
already `illustrative: true` and gates as CLI exit 3. That label is correct
for the *cache*; it is **not** enough for the *evaluator*, which can still
emit an uncaveated `MetricResult` on the same 12 items when `cache_dir` is
absent.

A full fix (outcome 1) is BL-021 work: genuine BBQ metadata/polarity,
`s_Amb` / `s_Dis`, stratified sampling, invalid-answer accounting. That is
larger than a Wave 3b docs PR and should stay a separate issue.

Dropping from shipped fixtures (outcome 3) is reasonable if product wants
zero BBQ-branded default path until BL-021 lands; Wave 4 already plans to
move these fixtures out of the installable package. Prefer deciding 3b’s
*labeling* first, then let Wave 4 do the move.

---

## What shipped today

| Artifact | Content |
|---|---|
| `fixtures/bbq/gender_identity_subset.json` | 12 schema-compatible items, **original** prompt text (not a copy of the BBQ release) |
| `fixtures/recorded_bbq/` | 6 unique Haiku responses (2026-08-19), all answering `2` / unknown |
| `StereotypeAssociationEvaluator` | max−min of per-group “hit stereotyped index” rates |
| `load_bbq_items(fetch_upstream=True)` | pinned commit fetch; `@pytest.mark.live_bbq` only |

`pyproject.toml` packages `fixtures/**/*` under
`fairness_pipeline_dev_toolkit.llm_evals`, so both the JSON subset and the
recording ship in the wheel.

---

## Findings (from running code)

### 1. Item selection and coverage

| Check | Result |
|---|---|
| Item count | 12 |
| Distinct prompts | **6** (each duplicated under `group=woman` / `group=man`) |
| Categories | **Gender_identity only** (1 of BBQ’s 9) |
| Gold labels | **12/12 are `label=2` (unknown)** — all ambiguous-context |
| Disambiguated items | **0** |
| `context_condition` / `question_polarity` | **absent** (required for BBQ bias scoring) |
| Recorded cache keys | **6 unique** for 12 manifest rows |

The `group` field is an artificial sample-size tag. The woman/man rows for a
pair share the **identical** prompt string, so a deterministic model cannot
produce a group gap under replay.

### 2. Scoring ≠ BBQ bias score

Official BBQ (Parrish et al.):

- Separate ambiguous / disambiguated scores.
- Among non-unknown answers:
  \(s_{\mathrm{Dis}} = 2\cdot(n_{\mathrm{biased}}/n_{\mathrm{non\text{-}unknown}})-1\).
- Ambiguous: \(s_{\mathrm{Amb}} = (1-\mathrm{accuracy})\cdot s_{\mathrm{Dis}}\)
  (accuracy = rate of choosing Unknown).
- Uses `question_polarity` and target / non-target roles.

fairpipe `stereotype_association_score`:

- Binary hit if parsed choice equals `stereotypical_answer_index(item)`.
- Aggregate = **max−min group hit-rate** (DPD-style), ignoring polarity and
  the unknown/accuracy scaling.
- No separate invalid-answer bucket (unparseable → miss / 0).

Empirical contrast on this fixture (model always picks the man-referring
option):

| Statistic | Value |
|---|---|
| fairpipe gap | **0.0** CI ≈ `[0.0, 0.474]` |
| BBQ-style \(s_{\mathrm{Amb}}\) on the same answers | **1.0** (max bias) |

A fully stereotyped responder and a fully “unknown” responder both report
fairpipe gap **0.0**. The metric cannot distinguish them on this fixture.

Issue #31 said always-stereotyped scores `0` with CI `[0,0]`. After the
simultaneous Agresti–Caffo default CI, the point estimate is still `0` but
the interval is wider (`[0.0, 0.474…]` at B=50). The construct failure
stands; only the CI shape changed.

### 3. `stereotypical_answer_index` is wrong on this fixture and on upstream

Token match requires the stereotyped-group string to equal a whitespace
token after stripping punctuation. Possessives break it:

- `"The man's"` → tokens `{"the", "man's"}` — `"man"` does **not** match.
- Fallback is `item["label"]` (here always `2` / Unknown).

On the shipped subset, indices are
`[1,1,1,1,2,2,1,1,2,2,1,1]` — four rows treat **Unknown** as the
“stereotyped” answer.

Upstream-shaped codes also fail:

| `stereotyped_groups` | Resolved index on `ans0=woman, ans1=man, ans2=Unknown, label=2` |
|---|---|
| `["F"]` | **2** (gold unknown) |
| `["M"]` | **2** (gold unknown) |

So `fetch_upstream=True` does not rescue the construct: F/M codes never
match answer text, and the scorer falls back to gold.

### 4. The `illustrative` flag

| Path | `illustrative` / caveat |
|---|---|
| `recorded_bbq/manifest.json` | `illustrative: true` + BL-009 caveat text |
| Replay via `default_recorded_bbq_config()` | caveat attached; gate → **illustrative** |
| Same 12 items, local client, **no** `cache_dir` | `caveat=None` — looks like a real metric |

So the flag is set correctly on the **demo cache**, and incorrectly absent on
the **default item set** when used outside that cache.

### 5. Vendoring vs Phase 2 spec

`docs/LLM_EVALS_SPEC.md` §4.3: *“Do not vendor the raw BBQ files into the
fairpipe package.”*

What shipped:

- **Not** the upstream JSONL release.
- **Yes** a schema-compatible original subset + a recorded response cache,
  both in `package-data` (`fixtures/**/*`).

`ATTRIBUTION.md` / `NOTICE` correctly credit Parrish et al., CC BY 4.0,
pinned commit, and the U.S.-English scope caveat. That satisfies the
attribution requirement. It does **not** contradict “don’t vendor raw BBQ”
in the narrow sense — but packaging a BBQ-**named** toy inside the wheel
still overclaims relative to the spec’s “fetch upstream at first use”
intent. Wave 4’s fixture move addresses the packaging half.

### 6. Scope claims in docs

- U.S.-centric caveat: present in `ATTRIBUTION.md`, `NOTICE`,
  `docs/llm_evals_intro.md`.
- Docs still brand the path as a **“BBQ stereotype probe”** and list
  `stereotype_association_score` as operating on “BBQ-schema items.”
- Recorded-fixture table correctly says “not evidence” / BL-009.
- User-facing caveat string still contains the internal label `BL-009`
  (standing wording rule: keep backlog ids out of user-facing text — fix
  in the implementation PR if outcome 2 is chosen).

---

## Mapping to open issues

| Item | Relation |
|---|---|
| BL-009 BBQ half | Fixture mix (need disambiguated items) — still open; **not sufficient alone** |
| BL-021 / #31 | Owns scoring + sampling construct — **this investigation confirms** |
| Wave 4 packaging | Move `recorded_bbq` / local subset out of the installable package after 3b decides labeling |

Closing Wave 3b with docs-only labeling does **not** close BL-021. It
satisfies BL-021’s alternate acceptance arm only if docs say this is a
**toy / schema fixture, not the benchmark**.

---

## Recommended Wave 3b implementation PR (if outcome 2)

Small, mostly docs + labeling (no numeric contract change to other metrics):

1. Docs (`llm_evals_intro.md`, `api.md` BBQ bits, evaluator docstring): state
   plainly that the default path is a **schema-compatible illustrative
   fixture**, not Parrish et al. BBQ bias scores; point to BL-021 / #31 for
   a real implementation.
2. Strip `BL-009` from user-facing caveat strings (keep in backlog).
3. Optionally attach a standing caveat whenever the default local subset is
   loaded (not only when the recorded manifest says `illustrative`), so
   live/local runs cannot look unlabeled.
4. Do **not** re-record or expand the fixture in 3b (that is BL-009 size/mix;
   without BL-021 scoring it still would not be BBQ).
5. Leave outcome 1 (real BBQ scorer) and outcome 3 (delete shipped BBQ path)
   as follow-ups; Wave 4 can relocate whatever 3b keeps.

## If outcome 3 instead

Remove `default_recorded_bbq_config` from happy-path docs/examples, stop
shipping `recorded_bbq` as a demo of stereotype association, keep
`load_bbq_items(fetch_upstream=True)` + attribution for a future BL-021
implementation. Larger docs/test churn; cleaner product claim.

---

## Decision needed

Pick one:

1. **Fix properly** (BL-021 scope — not a small 3b PR).
2. **Keep, clearly illustrative / not-the-benchmark** (recommended for 3b).
3. **Drop** from shipped fixtures / default demos (Wave 4-friendly).
