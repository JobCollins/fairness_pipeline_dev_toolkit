# fairpipe: LLM Evals — Feature Specification

**Status:** Draft for Cursor-assisted implementation
**Target version:** 0.10.0 (minor bump per `docs/VERSIONING.md` — additive capability, no breaking changes)
**Author:** Job Collins Dulo

---

## 1. Objective

Add LLM evaluation as a new capability inside fairpipe, so the same statistically-rigorous
measurement/mitigation/monitoring framework that fairpipe applies to classifiers and regressors
can be applied to outputs from LLMs and LLM-backed systems (e.g. generative summarization,
classification-via-prompting, RAG answers).

## 2. Scope decision (resolve before writing code)

"LLM evals" can mean two different things. Pick one as the v0.10.0 scope — don't build both at once.

| | **A. LLM Fairness Evals (recommended)** | **B. General LLM Quality Evals** |
|---|---|---|
| What it measures | Disparate treatment/output-quality across demographic groups in LLM behavior: counterfactual fairness, stereotype association, refusal-rate and toxicity disparity | Faithfulness, hallucination, relevance, groundedness — general answer quality |
| Fit with fairpipe identity | Direct extension of the existing "measurement, mitigation, monitoring" positioning and the Fairlearn/AIF360 comparison table | Overlaps with promptfoo, DeepEval, Ragas, OpenAI Evals — no differentiation, and it's a different product thesis |
| Reuses existing machinery | Yes — bootstrap CI, effect sizes, `min_group_size`, `Result` objects, monitoring/drift engine all apply directly | Partially — statistical rigor angle is weaker for single-answer quality scoring |

**Decision: build Option A.** It's the version of "LLM evals" that is actually a fairpipe feature
rather than a different product. Option B is an explicit **non-goal** for this phase (see §7).

## 3. Existing architecture Cursor must match

Before writing any code, read these in the repo and mirror their conventions — don't invent a
parallel style:

- `fairpipe/metrics/` — `FairnessAnalyzer`, `BaseMetric` (see `fairpipe.metrics.base.BaseMetric`),
  and how `Result` objects carry `.value`, `.ci`, `.effect_size`.
- `fairpipe/pipeline/` — YAML-configured transformer pipeline (`sensitive:`, `pipeline:` blocks,
  `load_config()`), and how `fairpipe pipeline` / `fairpipe run-pipeline` CLI commands consume it.
- `fairpipe/monitoring/` — `RealTimeFairnessTracker`, `ColumnMap`, `TrackerConfig`,
  `FairnessDriftAndAlertEngine`, `FairnessReportingDashboard`, `FairnessABTestAnalyzer`.
- `fairpipe/integration/` — `execute_workflow`, `pytest_plugin.assert_fairness`,
  `reporting.generate_validation_report`, `mlflow_logger.log_workflow_results`.
- CLI surface: `fairpipe validate`, `fairpipe pipeline`, `fairpipe run-pipeline`,
  `fairpipe train-regularized`, `fairpipe calibrate`, `fairpipe version`.
- Extras mechanism in `pyproject.toml`: `dev`, `training`, `monitoring`, `adapters`, `api`.
- Backend abstraction pattern: `FairnessAnalyzer(backend="native"|"fairlearn"|"aequitas")`.
- Statistical rigor as a hard requirement: bootstrap confidence intervals, effect sizes,
  `min_group_size` guards, intersectional analysis support.

The new capability should feel like it was built by the same team, not bolted on.

## 4. New capability design

### 4.1 Module and packaging
- New subpackage: `fairpipe/llm_evals/`
- New optional extra: `pip install fairpipe[llm]` (keeps `openai`/`anthropic`/HTTP client deps out
  of the core install, consistent with how `training`/`monitoring`/`adapters` are isolated today)
- New `BaseLLMEvaluator` class in `fairpipe/llm_evals/base.py`, structurally parallel to
  `BaseMetric` — `compute()` returns a `Result`-compatible object so downstream reporting,
  MLflow logging, and CI/CD gating work unmodified.

### 4.2 Provider abstraction
- `LLMClient` interface with adapters for at least OpenAI-compatible APIs and Anthropic's API,
  mirroring the existing `backend=` pattern (`provider="openai"|"anthropic"|"local"`).
- Credentials via environment variables only — never accepted as config/YAML values.
- All provider calls async + batched with retry/backoff (respect provider rate limits).

### 4.3 Core evaluators (Phase 1 minimum viable set)
1. **Counterfactual fairness probe** — templated prompts with a sensitive term swapped
   (name, gender, nationality, etc.); measure divergence in sentiment, toxicity, refusal rate,
   response length, and semantic similarity across the swapped variants. This is the flagship
   evaluator — directly analogous to `demographic_parity_difference` but for generative output.
2. **Refusal-rate disparity** — does the model refuse/hedge more often for some group-coded
   prompts than others.
3. **Toxicity/sentiment disparity** — score outputs with a pluggable classifier (moderation API
   or local model) and compare distributions across groups, with bootstrap CI exactly as
   `FairnessAnalyzer` does today.
4. **Stereotype association probe** — built on the **BBQ dataset** (Parrish et al., 2022;
   `github.com/nyu-mll/BBQ`), licensed **CC BY 4.0** (verified directly against the repo's
   `LICENSE` file — don't trust secondary citations, at least one paper misdescribes this as
   non-commercial-only). Its ambiguous/disambiguated question-pair structure is already a
   counterfactual comparison, which maps directly onto this evaluator's design. Implementation
   notes:
   - Do not vendor the raw BBQ files into the `fairpipe` package. Add a loader that fetches from
     the upstream repo (pinned to a specific commit for reproducibility) at install/first-use.
   - Add attribution (creator credit, CC BY 4.0 notice, link) in a `NOTICE`/`ATTRIBUTION.md` entry.
   - `docs/llm_evals_intro.md` must state plainly that BBQ's nine categories reflect U.S.
     protected-class categories and U.S. English social context — do not imply universal/
     cross-cultural coverage. Locale-adapted variants (precedent: PakBBQ, KoBBQ) are a possible
     future extension, explicitly out of scope for v0.10.0.
   - Do not use CrowS-Pairs or StereoSet as a basis for this evaluator — both have documented
     construct-validity and label-consistency problems in the fairness-NLP literature, which
     conflicts with the statistical-rigor standard the rest of fairpipe holds itself to.

All four should produce `Result` objects so they slot into the same reporting, CLI, and CI/CD
pipeline as existing metrics — no separate report format.

### 4.4 Config and CLI
- New YAML block `llm_eval:` alongside the existing `pipeline:`/`training:` blocks in
  `config.yml`, specifying provider, model, evaluators to run, and prompt template sets.
- New CLI command: `fairpipe llm-eval --config llm_eval.yml --report-md artifacts/llm_report.md`,
  following the exact flag conventions of `fairpipe validate`.
- `--dry-run` flag that estimates request count and approximate cost before making live calls —
  required, not optional, given probe suites can run hundreds of prompts.

### 4.5 Integration points
- `assert_llm_fairness()` in the pytest plugin, parallel to `assert_fairness()`.
- MLflow logger extended to log LLM eval results the same way `log_workflow_results` does today.
- CI/CD: extend `SvrusIO/fairpipe-action` with an `llm-fairness-check` mode.
- REST API: new endpoint under the existing `api` extra.
- Monitoring (Phase 3, not Phase 1): extend `RealTimeFairnessTracker` concept to sampled
  production LLM outputs.

## 5. Engineering constraints (non-negotiable)

- **No live network calls in unit tests.** Use recorded/mocked LLM responses as fixtures. Live
  integration tests go behind an explicit pytest marker (e.g. `@pytest.mark.live_llm`) that is
  opt-in and never run in default CI.
- **Response caching** keyed on (provider, model, prompt, params) so repeated eval runs during
  development don't re-bill the same requests.
- **Cost/rate controls**: `--dry-run`, configurable max-requests-per-run, exponential backoff on
  429s.
- **Content handling**: bias-probe transcripts can contain toxic or sensitive generated text by
  design. Do not dump raw transcripts into CI job summaries or default Markdown reports — reports
  should show aggregated metrics and CI intervals; raw transcripts go to a separate artifact file
  that isn't rendered automatically.

## 6. Deliverables

- `fairpipe/llm_evals/` module (base class, provider adapters, four Phase-1 evaluators, config
  schema)
- `fairpipe/integration/pytest_plugin.py` — `assert_llm_fairness()` addition
- `fairpipe/integration/mlflow_logger.py` — LLM eval logging support
- CLI: `llm-eval` subcommand
- Tests: unit tests against mocked responses (required for merge); live tests behind marker
  (optional, documented as opt-in)
- `case_studies/llm_counterfactual_fairness.ipynb` — a demonstration notebook in the same style
  as `case_studies/compas_racial_bias.ipynb`, run against a public benchmark subset, with a real
  measured effect size reported (not a placeholder number) — this is the credibility artifact,
  matching how the COMPAS/ACS notebooks ground the rest of the package's claims.
- `docs/llm_evals_intro.md` — a short explainer (new file, matching the existing style of
  `docs/getting_started.md`/`docs/integration_guide.md`) covering: what LLM fairness evals are,
  how they differ from the classifier/regressor metrics already in fairpipe, and a worked example
  using the new CLI command. Keep this scoped to Option A (§2) — don't describe general quality
  evals as if they're in scope.
- `DOCS.md` — new "Phase 8: LLM Fairness Evaluation" section, consistent with the existing
  Phase 1–7 structure.
- `docs/api.md` — API reference entries for the new module.
- `README.md` — add a row to the Fairlearn/AIF360/fairpipe comparison table (e.g.
  "LLM/GenAI fairness evals") since this is a genuine differentiator none of the compared tools
  have. Also add a **"Setting LLM provider credentials"** section (new, alongside the existing
  Install/Quick start sections) covering:
  - Which environment variable each supported provider reads (e.g. `OPENAI_API_KEY`,
    `ANTHROPIC_API_KEY`), and that `fairpipe[llm]` never reads credentials from YAML config or
    CLI flags — env vars only, per §5.
  - That no credentials are required for `provider="local"` (self-hosted/HF model path), so the
    section should make clear which usage paths do and don't need a key at all.
  - A pointer to `--dry-run` for estimating request volume/cost before a key gets used against a
    live provider.
  - This section is written in Phase 0 (where the provider abstraction and credential loading
    are built) and updated in later phases if new providers or auth paths are added — it should
    never fall behind what the code actually supports.
- `CHANGELOG.md` entry, `docs/VERSIONING.md`-consistent version bump to 0.10.0.

## 7. Explicit non-goals for this phase

- General-purpose LLM quality evaluation (faithfulness, hallucination, RAG groundedness,
  agentic tool-use correctness) — Option B from §2. Revisit only as a separately-scoped feature.
- Locale-adapted or non-English stereotype benchmarks (e.g. an Africa-context BBQ variant) — a
  plausible future differentiator, but a separate, deliberately-scoped effort, not v0.10.0.
- Production monitoring of live LLM traffic (Phase 3 at earliest, after Phase 1 ships and is used).
- A hosted/managed judge-model service — evaluators call the user's own configured provider.

## 8. Phased build plan

Every phase follows the same gate, in order: **Build → Test → Docs → Confirm → next phase.**
Docs (including `README.md`) are updated as part of the phase, immediately after its tests pass —
not deferred to the end of the project. Don't start Phase *n+1* until Phase *n*'s gate is closed.

### Phase 0 — scaffolding
**Build:** `LLMClient` provider abstraction (OpenAI-compatible + Anthropic adapters),
`BaseLLMEvaluator`, `llm_eval:` config schema, response caching layer.
**Test (run before moving on):**
- Unit tests for each `LLMClient` adapter against mocked HTTP responses — assert zero real
  network calls occur during the test run.
- `BaseLLMEvaluator` interface contract test — a minimal concrete evaluator's `compute()` returns
  a `Result`-compatible object.
- Config schema validation tests — at least one valid and one invalid `llm_eval:` block.
- Cache tests — hit/miss behavior, cache key composition (provider, model, prompt, params).
**Docs:** `README.md` "Setting LLM provider credentials" section (see §6) added. `docs/api.md`
stub entries for `LLMClient`/`BaseLLMEvaluator`. `CHANGELOG.md` entry for the scaffolding.

### Phase 1 — flagship evaluator
**Build:** Counterfactual fairness probe end-to-end (config → CLI → `Result` → Markdown report),
`--dry-run` cost/request estimate, one case study notebook.
**Test:**
- Counterfactual probe unit test against a mocked-response fixture with a known, engineered
  divergence — assert the computed metric and CI match the expected value within tolerance.
- CLI end-to-end test: `fairpipe llm-eval` run against a mocked provider produces a valid
  Markdown report file.
- `--dry-run` test: asserts a request-count/cost estimate is produced and no live call is made.
- Case study notebook executes end-to-end without error (e.g. via `nbclient`/`papermill` in CI)
  and reports a real, reproducible effect size — not a placeholder — matching the standard set by
  the COMPAS notebook's 0.2116 EOD finding.
**Docs:** `docs/llm_evals_intro.md` added. `DOCS.md` gets its new "Phase 8: LLM Fairness
Evaluation" section (covering the counterfactual evaluator only, at this point). `docs/api.md`
updated with the working CLI command and config block. `README.md` comparison-table row added.
`CHANGELOG.md` entry.

### Phase 2 — remaining evaluators
**Build:** Refusal-rate disparity, toxicity/sentiment disparity, BBQ-based stereotype association
probe (with the loader/attribution/scope-caveat requirements from §4.3.4), pytest plugin
(`assert_llm_fairness()`), MLflow logging.
**Test:**
- Unit test per new evaluator against mocked/fixture responses with known expected output.
- BBQ loader test against a small local fixture subset (default CI) — separate, opt-in
  `@pytest.mark.live_llm`-style marker for a test that actually pulls from the pinned upstream
  commit.
- Attribution/license file existence and content check (`NOTICE`/`ATTRIBUTION.md` mentions BBQ,
  CC BY 4.0, and the upstream link).
- `assert_llm_fairness()` tests — raises on threshold violation, passes within threshold, mirrors
  the existing `assert_fairness()` test pattern.
- MLflow logging test — asserts LLM eval results are logged the same way
  `log_workflow_results` logs today (local tracking URI in tests, not a live MLflow server).
**Docs:** `DOCS.md` Phase 8 section extended to cover all four evaluators. `docs/api.md` updated.
`README.md` "Setting LLM provider credentials" section revisited if Phase 2 introduces any new
provider or auth path (e.g. a moderation API key for toxicity scoring). `CHANGELOG.md` entry.

### Phase 3 — production surface
**Build:** CI/CD Action (`llm-fairness-check` mode), REST API endpoint, monitoring integration
(sampled production LLM outputs feeding the existing drift/tracker stack).
**Test:**
- REST API endpoint tests via a test client against a mocked LLM backend (request/response
  contract, error handling).
- CI/CD Action entrypoint test — config parsing and exit-code behavior against a local harness
  (no real GitHub Actions run needed).
- Monitoring integration tests — sampling logic, aggregation, and that drift alerts fire on
  synthetic LLM-output data the same way they do for classifier output today.
**Docs:** `docs/integration_guide.md` updated with the REST API endpoint and CI/CD Action mode.
`README.md` CI/CD section updated with the `llm-fairness-check` example (mirroring the existing
`fairness-check` GitHub Action example). `CHANGELOG.md` entry, version bump to 0.10.0 finalized
per `docs/VERSIONING.md`.

## 9. Definition of done (applies to every phase)

- All of that phase's tests pass, with no live network calls in the default (non-`live_llm`) run.
- `Result` objects from new evaluators are consumable by existing reporting/MLflow/CI-gating code
  without special-casing.
- Every doc listed under that phase's "Docs" step above is updated in the same commit as the code
  — including `README.md` where the phase touches it. Docs are not a follow-up task.
- The phase is demonstrably runnable end-to-end (not just unit-tested in isolation) before the
  next phase starts.
