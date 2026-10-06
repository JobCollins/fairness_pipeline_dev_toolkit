# Package and import identity

**PyPI name:** [`fairpipe`](https://pypi.org/project/fairpipe/)

**Public import:** `import fairpipe` (documented)

**Compatibility import:** `import fairness_pipeline_dev_toolkit` — still supported this
release with no deprecation. Both namespaces re-export the same objects (object
identity preserved via the `fairpipe` shim).

| Role | URL |
|---|---|
| Canonical development repo | https://github.com/JobCollins/fairness_pipeline_dev_toolkit |
| Public mirror (publishes to PyPI) | https://github.com/SvrusIO/fAIr |
| Docs site | https://svrusio.github.io/fAIr/ |

## Documented modules that exist

Import these under either namespace (`fairpipe.*` or `fairness_pipeline_dev_toolkit.*`
where a shim exists):

| Module | Notes |
|---|---|
| `fairpipe` / `fairness_pipeline_dev_toolkit` | Top-level public API |
| `fairpipe.metrics` | `FairnessAnalyzer`, `MetricResult`, adapters |
| `fairpipe.io` | `load_data` (also re-exported on `fairpipe`) |
| `fairpipe.pipeline` | Orchestration, transformers |
| `fairpipe.pipeline.config` | `PipelineConfig`, `load_config`, `find_config_file` |
| `fairpipe.stats` | Bootstrap / Bayesian / effect-size helpers |
| `fairpipe.integration` | Reporting, gating, workflow helpers |
| `fairpipe.llm_evals` | LLM fairness evals (`[llm]` for live providers) |
| `fairpipe.exceptions` | Toolkit exception types |
| `fairpipe.training` | Optional `[training]` |
| `fairpipe.monitoring` | Optional `[monitoring]` |
| `fairpipe.api` | Optional `[api]` |
| `fairpipe.measurement` | Measurement / tracking helpers (`[tracking]` for MLflow) |

Typing: both packages ship `py.typed`. Prefer the `fairpipe` import in new code.
