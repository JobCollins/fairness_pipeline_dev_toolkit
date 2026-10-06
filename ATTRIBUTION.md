# Third-party attribution

## BBQ (Bias Benchmark for QA)

- **Dataset:** BBQ (Bias Benchmark for QA), Parrish et al., 2022
- **License:** CC BY 4.0
- **Upstream:** https://github.com/nyu-mll/BBQ
- **Pinned commit:** `bea11bd97d79217245b5871acd247b9d6eb24598`

fairpipe keeps a **schema-compatible local subset** for tests
(`fairness_pipeline_dev_toolkit/llm_evals/fixtures/bbq/gender_identity_subset.json`).
That subset is original prompt text written to match BBQ's JSON schema; it is **not** a
copy of the full BBQ release and is **not** loaded by default. Pass the path explicitly
or use `load_bbq_items(fetch_upstream=True)` (gated behind `@pytest.mark.live_bbq`).

`stereotype_association_score` is an experimental max−min stereotyped-answer rate on
BBQ-schema items. It is **not** the published BBQ bias score
([issue #31](https://github.com/JobCollins/fairness_pipeline_dev_toolkit/issues/31) /
BL-021).

**Scope caveat:** BBQ items encode U.S. English social stereotypes. Results are not a
general-purpose worldwide bias audit.
