# fairpipe — Technical Backlog
**Document ID:** BACKLOG-001
**Version:** 1.0
**Created:** 2026-05-13
**Owner:** Svrus LLC
**Source:** Issues identified during COMPAS recidivism case study development

---

## Summary

| ID | Title | Priority | Target Version |
|----|-------|----------|---------------|
| BL-001 | `fairpipe validate` missing `--threshold` flag | P0 | v0.7.3 |
| BL-002 | `execute_workflow` fails on mixed-type DataFrames | P0 | v0.7.3 |
| BL-003 | `execute_workflow` does not apply sample weights to model training | P0 | v0.7.3 |
| BL-004 | `apply_pipeline` returns weights in opaque tuple — easy to discard silently | P1 | v0.8.0 |
| BL-005 | CLI log output pollutes stdout alongside report content | P1 | v0.7.3 |
| BL-006 | `test_risk_ratio_identity` Hypothesis flakiness under float edge cases | P2 | backlog |
| BL-007 | Expand LLM counterfactual recorded-cache fixture to clear `min_group_size=5` | P1 | v0.8.0 |
| BL-008 | Phase 2 LLM evaluators: per-evaluator recorded-cache fixtures (≥5/group) | P1 | v0.8.0 |

---

## BL-001 — `fairpipe validate` missing `--threshold` flag

### Where Discovered
Attempting to run `fairpipe validate --threshold 0.05` in the COMPAS case study notebook.
The CLI rejected the flag with `error: unrecognized arguments: --threshold 0.05`.

### Impact
**High.** The `--threshold` flag is the most important missing CLI feature. Without it,
the CLI cannot produce a pass/fail verdict — it always exits 0 regardless of how severe
the bias is. This directly undermines the CI/CD use case, which is fairpipe's core value
proposition. Any team trying to use `fairpipe validate` in a GitHub Actions workflow
cannot enforce a fairness threshold from the command line.

### Current Workaround
Threshold comparison must be done in Python:
```python
from fairpipe import FairnessAnalyzer
dpd = analyzer.demographic_parity_difference(with_ci=True)
passed = dpd.value <= THRESHOLD
```

### Long-Term Fix
Add `--threshold` as an optional flag to the `validate` CLI command with a sensible
default of `0.05`. The command should exit `1` when the primary metric (DPD by default)
exceeds the threshold, and exit `0` when it passes.

```
fairpipe validate \
  --csv data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --threshold 0.05 \
  --metric equalized_odds_difference
```

Also add a `--metric` flag so users can choose which metric to evaluate against the
threshold (default: `demographic_parity_difference`). This is important because EOD
is often the more appropriate metric in high-stakes domains.

**Implementation location:** `fairness_pipeline_dev_toolkit/cli/` — `validate` command handler.

**Acceptance criteria:**
- `fairpipe validate --threshold 0.05` exits `1` when DPD > 0.05
- `fairpipe validate --threshold 0.05` exits `0` when DPD ≤ 0.05
- `fairpipe validate` with no threshold flag still exits `0` (backward compatible)
- The validation report includes the threshold and pass/fail status when `--threshold` is provided
- The GitHub Action `fairpipe-action` can be updated to use this flag directly
  rather than parsing the exit code from the Python API

---

## BL-002 — `execute_workflow` fails on mixed-type DataFrames

### Where Discovered
Calling `execute_workflow(config=config, df=df_bw, ...)` with the raw COMPAS DataFrame,
which contains string columns (names, dates, case numbers). The orchestrator passed the
entire DataFrame to `LogisticRegression.fit()`, which raised:

```
ValueError: could not convert string to float: 'deandrae counts'
```

Traceback in `fairness_pipeline_dev_toolkit/integration/orchestrator.py:134`.

### Impact
**Critical.** Real-world datasets always contain string columns, identifiers, and dates.
`execute_workflow` is fairpipe's flagship end-to-end function and the primary demo
feature, but it is unusable on any real dataset without manual preprocessing. This is
the first thing a new user will try and the first thing that will fail. It produces a
cryptic sklearn error rather than a helpful fairpipe error message.

### Current Workaround
Users must manually select only numeric features before calling `execute_workflow`,
or bypass it entirely and call `apply_pipeline` + a manual model training loop (as
done in the COMPAS notebook).

### Long-Term Fix
**Part 1 — Auto feature selection (fallback behaviour):**

Add a `_prepare_features()` utility in the orchestrator that automatically selects
numeric columns, excluding the target and sensitive attribute columns:

```python
def _prepare_features(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str | list[str],
) -> pd.DataFrame:
    """
    Auto-select numeric columns suitable for model training.
    Excludes target, sensitive attribute, and all non-numeric columns.
    Raises ValueError with a helpful message if no features remain.
    """
    exclude = set(
        [target_col] +
        (sensitive_col if isinstance(sensitive_col, list) else [sensitive_col])
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found after excluding target and sensitive "
            f"columns. Either pass a pre-processed DataFrame or specify 'features' "
            f"in your config YAML. Available columns: {list(df.columns)}"
        )
    return df[feature_cols]
```

**Part 2 — Explicit feature specification in config (preferred behaviour):**

Add an optional `features` key to `PipelineConfig` and the YAML config schema:

```yaml
# Optional — if omitted, auto-selection is used
features: ["age", "priors_count", "juv_fel_count", "juv_misd_count"]
```

```python
# In fairness_pipeline_dev_toolkit/pipeline/config.py
@dataclass
class PipelineConfig:
    sensitive: list[str]
    pipeline: list[dict]
    features: list[str] | None = None   # ← add this field
    training: dict | None = None
    fairness_metric: str = "demographic_parity_difference"
    validation_threshold: float = 0.05
```

When `features` is specified, use exactly those columns and raise a clear `ValueError`
if any are missing. When absent, fall back to auto-selection and log a warning:

```
WARNING: No 'features' specified in config. Auto-selecting numeric columns: 
['age', 'priors_count', 'juv_fel_count', ...]. Specify 'features' in your 
config for reproducible results.
```

**Implementation location:** `fairness_pipeline_dev_toolkit/integration/orchestrator.py`
— `run_baseline_measurement()` and the transform+train step.

**Acceptance criteria:**
- `execute_workflow` runs successfully on the raw COMPAS DataFrame without any
  manual preprocessing
- When `features` is specified in config, only those columns are used
- When `features` is absent, numeric columns are auto-selected with a logged warning
- A clear `ValueError` is raised (not a sklearn error) when no numeric features remain
- All existing tests pass

---

## BL-003 — `execute_workflow` does not apply sample weights to model training

### Where Discovered
Running `execute_workflow` after `InstanceReweighting` mitigation produced zero change
in fairness metrics. Investigation revealed that `apply_pipeline` returns a tuple
`(DataFrame, metadata_dict)` where `metadata['sample_weight']` contains the weights,
but the orchestrator discards the second return value with `_`:

```python
# Current code in orchestrator — weights silently discarded
df_transformed, _ = apply_pipeline(pipeline, df)
model.fit(X, y)  # no sample_weight passed
```

### Impact
**Critical.** This is a silent bug — `execute_workflow` appears to run successfully
and returns metrics, but the mitigation has had zero effect on the model. A user
who relies on `execute_workflow` to apply `InstanceReweighting` is getting a false
sense of mitigation. The before and after metrics will be identical.

### Current Workaround
Manually unpack `apply_pipeline` and pass weights to `model.fit()`:
```python
df_mitigated, metadata = apply_pipeline(pipeline, df_bw)
sample_weights = metadata.get("sample_weight", None)
clf_fair.fit(X, y, sample_weight=sample_weights)
```

### Long-Term Fix
**Immediate fix (v0.7.3):**

In `orchestrator.py`, unpack the metadata and pass weights to `fit()`:

```python
# Fix in run_baseline_measurement or equivalent transform+train step
df_transformed, metadata = apply_pipeline(pipeline, df)
sample_weight = metadata.get("sample_weight", None)

if sample_weight is not None:
    logger.info(f"Applying sample weights from pipeline metadata. "
                f"Weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")

model.fit(X_train, y_train, sample_weight=sample_weight[train_idx] if sample_weight is not None else None)
```

Note: the sample weights must be sliced to match the training set indices after
`train_test_split` — do not pass the full weight array when training on a subset.

**Related fix:** This bug and BL-002 should be fixed together in the same PR since
they both modify `run_baseline_measurement()` in the same file.

**Implementation location:** `fairness_pipeline_dev_toolkit/integration/orchestrator.py`

**Acceptance criteria:**
- `execute_workflow` with `InstanceReweighting` produces measurably different
  before/after metrics
- Sample weights are correctly sliced to match the training split
- A log message confirms weights were applied
- The COMPAS notebook Cell 9 produces a non-zero improvement when using
  `execute_workflow` instead of the manual training loop

---

## BL-004 — `apply_pipeline` returns weights in opaque tuple — easy to discard silently

### Where Discovered
`apply_pipeline` returns `(DataFrame, dict)`. The weights are in `result[1]['sample_weight']`.
Any developer naturally writing `df_transformed, _ = apply_pipeline(...)` silently
discards the weights — exactly what the orchestrator was doing (BL-003).

### Impact
**Medium — API design debt.** The current return type is a footgun. There is no type
hint, no IDE autocomplete, and no indication that the second return value contains
critical information. This will cause the BL-003 bug to recur for any developer
who calls `apply_pipeline` directly.

### Current Workaround
Document the return type explicitly in the docstring and add a type hint:
```python
def apply_pipeline(pipeline, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
        tuple: (transformed_df, metadata) where metadata may contain:
            - 'sample_weight': np.ndarray of per-sample weights (if reweighting was applied)
    """
```
This is the minimum acceptable fix for v0.7.3.

### Long-Term Fix (v0.8.0)
Replace the bare tuple return with a typed `PipelineResult` dataclass. This is a
**breaking change** and requires a version bump and deprecation notice:

```python
# fairness_pipeline_dev_toolkit/pipeline/results.py
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class PipelineResult:
    """
    Return type for apply_pipeline().
    
    Attributes:
        data: The transformed DataFrame.
        sample_weight: Per-sample weights produced by reweighting transformers.
            Pass to model.fit(sample_weight=result.sample_weight) during training.
            None if no reweighting transformer was applied.
        transformers_applied: Names of transformers that were applied.
        metadata: Additional transformer-specific outputs.
    """
    data: pd.DataFrame
    sample_weight: np.ndarray | None = None
    transformers_applied: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

Update `apply_pipeline` to return `PipelineResult`:
```python
def apply_pipeline(pipeline, df: pd.DataFrame) -> PipelineResult:
    ...
    return PipelineResult(
        data=df_transformed,
        sample_weight=metadata.get("sample_weight"),
        transformers_applied=[t.name for t in pipeline.transformers],
        metadata=metadata,
    )
```

All callers — including the orchestrator — must be updated. The old tuple return
should be deprecated with a warning in v0.7.x and removed in v0.8.0.

**Acceptance criteria:**
- `apply_pipeline` returns `PipelineResult`
- IDE autocomplete shows `.sample_weight`, `.data`, `.transformers_applied`
- `result.sample_weight` is `None` when no reweighting transformer ran
- Backward compatibility shim emits `DeprecationWarning` if tuple unpacking is used
- All existing tests updated to use new return type

---

## BL-005 — CLI log output pollutes stdout alongside report content

### Where Discovered
Running `fairpipe validate` in the COMPAS notebook produced output mixing INFO log
lines with the markdown report:

```
2026-05-13 17:48:00,153 - fairness_pipeline_dev_toolkit.cli - INFO - CLI started
2026-05-13 17:48:00,155 - fairness_pipeline_dev_toolkit.cli - INFO - Starting validation
# Fairness Validation Report (CLI)
...
2026-05-13 17:48:00,580 - fairness_pipeline_dev_toolkit.cli - INFO - Validation completed
```

### Impact
**Medium.** When the CLI is used in a notebook, piped to a file, or consumed
programmatically, the log lines contaminate the output. The markdown report
cannot be cleanly extracted without post-processing. This is particularly
problematic for the GitHub Action, which parses the report output.

### Current Workaround
Redirect stderr in the shell: `fairpipe validate ... 2>/dev/null`

### Long-Term Fix
**Part 1 — Send logs to stderr, not stdout (immediate):**

Ensure all CLI log handlers write to `sys.stderr`. Report content (markdown, JSON)
writes to `sys.stdout`. This is standard Unix convention and means `2>/dev/null`
cleanly separates logs from output.

**Part 2 — Add `--quiet` flag:**

```
fairpipe validate --csv data.csv ... --quiet
```

When `--quiet` is set, suppress all INFO and WARNING log output. Only ERROR level
messages are shown. The report is still written to `--out` if specified.

**Part 3 — Respect non-TTY context:**

If stdout is not a TTY (i.e., output is being piped or redirected), automatically
suppress INFO logs without requiring `--quiet`. This is the behaviour of tools like
`git`, `curl`, and `pip`:

```python
import sys
if not sys.stdout.isatty():
    logging.getLogger("fairness_pipeline_dev_toolkit").setLevel(logging.ERROR)
```

**Implementation location:** `fairness_pipeline_dev_toolkit/cli/__init__.py` or
the CLI entry point where the logger is configured.

**Acceptance criteria:**
- `fairpipe validate ... | cat` produces only the markdown report, no log lines
- `fairpipe validate ... --quiet` suppresses all INFO output
- `fairpipe validate ...` in an interactive terminal still shows INFO logs
- The GitHub Action `fairpipe-action` no longer needs `2>/dev/null` to clean output

---

## BL-006 — `test_risk_ratio_identity` Hypothesis flakiness (follow-up, non-blocking)

### Where Discovered
Full-suite runs with warnings enabled (`pytest -W default`) intermittently fail on
`tests/property_based/test_property_based.py::TestEffectSizeProperties::test_risk_ratio_identity`.
Hypothesis falsifying example: `rate1=1.544296017645972e-10`, `rate2=1e-10`. Observed during
LLM evals Phase 0/1 validation; **not introduced by LLM evals**.

### Impact
**Low (test infra).** Default CI uses `--disable-warnings` and may not hit this every run, but
the property test guard is unsound for denormal float edge cases near `1e-10`.

### Root cause (likely)
The "equal rates" precondition uses a tolerance that can treat distinct tiny rates as equal while
`risk_ratio()` returns a value far from 1.0 due to floating-point division.

### Suggested fix (later triage)
- Tighten equality precondition (`math.isclose` aligned with `risk_ratio` semantics), **or**
- Constrain Hypothesis `min_value` away from denormals, **or**
- Skip identity assertion below a stable rate floor.

**Acceptance criteria:**
- `test_risk_ratio_identity` passes reliably across 100+ Hypothesis examples with `-W default`
- No production `risk_ratio()` change unless a real numeric bug is confirmed

---

## BL-007 — Expand LLM counterfactual recorded-cache fixture to clear `min_group_size=5`

### Where Discovered
Phase 1 gate review of `case_studies/llm_counterfactual_fairness.ipynb`. The committed
Anthropic cache replay fixture has **n=1 prompt per demographic group** (`woman`, `man`,
`nonbinary`). Shared LLM eval guard now mirrors classifier semantics: groups below
`DEFAULT_LLM_MIN_GROUP_SIZE=5` are excluded and the metric returns **`nan`**.

### Impact
**High (credibility).** The notebook is the primary artifact readers use to judge LLM fairness
evals. The illustrative `allow_small_samples=True` smoke test proves plumbing works but must
not be mistaken for a model behavior finding. Phase 2 evaluator scaffolding can proceed; this
item tracks the **production-grade case study** separately.

### Scope
**Counterfactual probe only** (`fixtures/recorded_counterfactual/`). Phase 2 evaluators
require the same cache-once-replay treatment under separate fixtures — see **BL-008** so
that work is not rediscovered per evaluator during Phase 2 implementation.

### Target fixture shape
At minimum **5 prompts per group** for the canonical gender dimension, e.g.:

- Multiple `{name}` defaults (≥5 distinct names) with the same hiring template, **or**
- Multiple role templates × names such that each group accumulates ≥5 provider calls

Re-record live via `@pytest.mark.live_llm` → `populate_recorded_counterfactual_cache()`,
commit cache + manifest, regenerate notebook divergence (with CI once n supports it).

### Acceptance criteria
- Each group in `fixtures/recorded_counterfactual/` has **≥5** cached responses
- `run_llm_eval(default_recorded_counterfactual_config())` returns a **finite** metric at
  default `min_group_size=5` **without** `allow_small_samples`
- Notebook updated to drop illustrative override; reports production-threshold result
- `tests/llm_evals/test_recorded_cache.py` asserts finite metric at default threshold

---

## BL-008 — Phase 2 LLM evaluators: per-evaluator recorded-cache fixtures (≥5/group)

### Where Discovered
Phase 1 gate close. BL-007 covers the counterfactual probe fixture only. Phase 2 adds three
evaluators that share the same `llm_evals` runner, `ResponseCache`, replay-only client, and
`DEFAULT_LLM_MIN_GROUP_SIZE=5` guard — each will need its **own** committed recorded-cache
fixture sized to clear the default threshold without `allow_small_samples`.

### Evaluators requiring fixtures (Phase 2)
| Evaluator | Suggested fixture path (pattern) |
|-----------|----------------------------------|
| `refusal_rate_disparity` | `fixtures/recorded_refusal/` |
| `toxicity_sentiment_disparity` | `fixtures/recorded_toxicity/` |
| `stereotype_association_score` (BBQ probe) | `fixtures/recorded_bbq/` |

Follow the Phase 1 pattern established by `recorded_counterfactual.py`:
- `populate_*_cache()` live behind `@pytest.mark.live_llm`
- Commit `cache/*.txt` + `manifest.json`
- `default_recorded_*_config()` for zero-API replay in tests/notebook
- Default-path tests assert **finite** metrics at `min_group_size=5` (no illustrative override)

### Impact
**Medium–high (Phase 2 velocity).** Without pre-planned fixtures, each evaluator will either
ship synthetic demos (credibility risk) or block CI on live API keys. Planning fixtures up
front keeps Phase 2 scaffolding separate from credibility artifacts.

### Acceptance criteria (per evaluator)
- ≥5 provider responses per demographic group in committed cache
- `run_llm_eval(default_recorded_*_config())` finite at default `min_group_size=5`
- `tests/llm_evals/test_recorded_*_cache.py`: replay test + `@pytest.mark.live_llm` populate hook
- Document fixture regeneration in `docs/llm_evals_intro.md`

---

## Implementation Order

Given the conference deadline (May 19) and the importance of a working end-to-end
demo, the recommended implementation order is:

| Order | Issues | Rationale |
|-------|--------|-----------|
| 1st | BL-002 + BL-003 | Fix together — same file, same function. Makes `execute_workflow` work on real data with real mitigation. Enables Option A in the notebook. |
| 2nd | BL-001 | Adds `--threshold` to CLI. Unlocks clean CI/CD demo and unblocks the GitHub Action's native threshold support. |
| 3rd | BL-005 | Cleans up output. Low risk, high polish. |
| Post-conference | BL-004 | Breaking API change. Needs deprecation cycle. Target v0.8.0. |

---

## GitHub Issues to Create

Create one GitHub issue per backlog item. Suggested labels:

| Issue | Labels |
|-------|--------|
| BL-001 | `enhancement`, `cli`, `ci-cd` |
| BL-002 | `bug`, `execute_workflow`, `good first issue` |
| BL-003 | `bug`, `execute_workflow`, `mitigation` |
| BL-004 | `enhancement`, `api-design`, `breaking-change` |
| BL-005 | `enhancement`, `cli`, `developer-experience` |
| BL-006 | `bug`, `testing`, `hypothesis`, `good first issue` |
| BL-007 | `enhancement`, `llm-evals`, `case-study`, `documentation` |
| BL-008 | `enhancement`, `llm-evals`, `phase-2`, `testing` |

---

*Document ID: BACKLOG-001 | Version: 1.0 | Created: 2026-05-13 | Owner: Svrus LLC*
*All issues sourced from COMPAS case study development session, May 13, 2026.*
