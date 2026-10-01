# AGENTS.md — fairpipe

Instructions for AI coding agents (GitHub Agentic Workflows, Copilot, Claude Code,
Cursor, Codex) working in this repository. Humans are welcome to read it too.
Keep it short, factual, and current: if a command or path here goes stale, fix
this file in the same PR.

## What this project is

`fairpipe` is a Python library for measuring, mitigating, and monitoring
fairness in ML systems (classical models and LLMs). Users rely on its numbers
to make claims about bias, so **a silently wrong metric is worse than a crash**.
Correctness of the statistics comes before speed, style, or cleverness.

## Repository map

| Path | What lives there |
|---|---|
| `fairness_pipeline_dev_toolkit/` | **The real source.** All code changes go here. |
| `fairpipe/` | Thin compatibility shim that re-exports `fairness_pipeline_dev_toolkit`. Do not put logic here. If you add a public name to a source `__init__.py` `__all__`, the shim picks it up automatically — check `tests/test_namespace.py` still passes. |
| `fairness_pipeline_dev_toolkit/metrics/` | Core fairness metrics + adapters (native, fairlearn, aequitas). `registry.json` maps metric names. |
| `fairness_pipeline_dev_toolkit/stats/` | Bootstrap CIs, Bayesian estimates, small-group handling. |
| `fairness_pipeline_dev_toolkit/pipeline/` | Bias detectors, transformers (e.g. `ReweighingTransformer`), orchestration, YAML config. |
| `fairness_pipeline_dev_toolkit/training/` | Fairness-aware training (sklearn reductions, regularized / Lagrangian NN, calibration). Torch is optional (`[training]`). |
| `fairness_pipeline_dev_toolkit/monitoring/` | Drift, tracking, A/B tests, dashboards (`[monitoring]`). |
| `fairness_pipeline_dev_toolkit/llm_evals/` | LLM fairness evals (`[llm]`): probes, evaluators, BBQ, cached fixtures, live-call kill switch. |
| `fairness_pipeline_dev_toolkit/api/` | FastAPI REST service (`[api]`). |
| `fairness_pipeline_dev_toolkit/cli/` | `fairpipe` CLI entry point (`cli/main.py`). |
| `tests/` | pytest suite, mirrors the package layout. |
| `benchmarks/` | Performance benchmarks (100k-row metrics, pipeline, bootstrap). |
| `case_studies/` | Published notebooks (COMPAS, ACS, LLM). Outputs are intentionally kept. |
| `docs/` | Sphinx docs, ADRs, specs (`LLM_EVALS_SPEC.md`), backlog (`fairpipe-technical-backlog.md`), `VERSIONING.md`. |

## Setup and commands

Python 3.10, 3.11 and 3.12 are supported; CI runs all three on Ubuntu, macOS and
Windows.

```bash
pip install -e ".[dev,api]" -r requirements.txt   # same as CI
```

Run these before proposing any change. They are exactly what CI enforces:

```bash
black --check fairness_pipeline_dev_toolkit tests          # black==25.9.0, line length 100
flake8 fairness_pipeline_dev_toolkit                       # config in setup.cfg
pytest --cov=fairness_pipeline_dev_toolkit --cov-report=term-missing
coverage report --fail-under=85                            # coverage gate: 85%
```

Faster loops while iterating:

```bash
pytest tests/metrics -q                     # one area
pytest tests/test_metrics_core.py -k parity  # one test
```

Fairness smoke gate (also in CI):

```bash
python scripts/quick_fairness_check.py --csv ci_sample.csv \
  --y-true y_true --y-pred y_pred --sensitive group --threshold 0.12 --min-group-size 20
```

Benchmarks (use these to justify any performance change):

```bash
python benchmarks/benchmark_metrics_100k.py
python benchmarks/benchmark_pipeline.py
python benchmarks/benchmark_bootstrap.py
```

## Hard rules (never break these)

1. **No live LLM calls.** Live provider HTTP is forbidden by default and guarded by
   the `FAIRPIPE_LLM_ALLOW_LIVE` kill switch (`LiveLLMCallForbidden`). Never set
   that variable, never add code that bypasses the guard, never remove the
   autouse fixture in `tests/conftest.py`. Tests marked `live_llm` / `live_bbq`
   are excluded by default and must stay that way. Use cached fixtures in
   `llm_evals/fixtures/`.
2. **Never weaken a gate to make CI pass.** Do not lower the coverage threshold,
   loosen fairness thresholds, add `# noqa` / `# type: ignore` / `pytest.skip`
   to hide a real problem, or delete a failing test. Fix the cause or explain
   why you cannot.
3. **Numerical behaviour is a contract.** A refactor or optimisation must
   produce the same metric values (exact, or within a stated `np.allclose`
   tolerance with a test proving it). If a change *intends* to alter a number,
   say so explicitly, add a test pinning the new value, and add a CHANGELOG
   entry under "Upgrade notes".
4. **Public API is stable.** Anything in an `__init__.py` `__all__`, CLI flags,
   YAML config keys, REST request/response fields, and result-dict keys are
   public. Renames need a deprecation alias + `FutureWarning` for one release
   (see the `counterfactual_fairness_*` → `demographic_swap_*` rename in
   `CHANGELOG.md` as the house pattern). Follow `docs/VERSIONING.md`.
5. **Optional dependencies stay optional.** `torch`, `fairlearn`, `aequitas`,
   `openai`, `anthropic`, `fastapi`, `streamlit`, `dash` must only be imported
   inside the extra that needs them, with a clear error if missing. `import
   fairpipe` must work on a bare `pip install fairpipe`.
6. **Don't touch secrets, release, or publishing.** Do not edit
   `scripts/publish.sh`, `.github/workflows/release.yml`, version numbers in
   `pyproject.toml`, or anything that handles tokens, unless the task is
   explicitly about that.
7. **Datasets and licences.** BBQ is CC BY 4.0 — keep attribution in
   `ATTRIBUTION.md`. Do not add new datasets without a licence note.

## Fairness and statistics checklist

When code touches metrics, stats, detectors, transformers or evaluators, check:

- **Group edge cases:** a group with zero rows, one row, all-positive or
  all-negative labels, or `NaN` as a group value. Divide-by-zero must produce a
  documented result (`nan`, a warning, or an error) — never a silent `0`.
- **Direction and sign:** differences vs ratios, which group is the reference,
  absolute vs signed values. Docstrings must state the convention.
- **Min group size:** small groups must go through the existing
  `min_group_size` / small-sample handling, not bypass it.
- **Intersectional groups:** combined sensitive attributes must not drop rows
  with missing values silently.
- **Randomness:** bootstrap and sampling code takes a `random_state` / `rng`
  and is reproducible with a fixed seed.
- **Weights:** sample weights from transformers must be finite, positive, and
  aligned with row order after any reindexing.
- **Leakage:** fit on train only; `transform` must not re-learn from test data.

## Code style

- Formatting is black (line length 100) and isort (profile black). Lint is flake8.
- Type hints on all new public functions; numpy-style docstrings with a short
  example for anything public.
- Raise toolkit exceptions from `fairness_pipeline_dev_toolkit/exceptions.py`
  rather than bare `Exception`. Use `raise ... from err` inside `except`.
- Prefer vectorised pandas/numpy over Python loops on rows. Don't add a new
  dependency for something numpy/pandas already does.
- Keep functions small; avoid editing unrelated code in the same change.

## Tests

- Every bug fix gets a regression test that fails before the fix.
- Every new public function gets unit tests, including the edge cases above.
- Tests must be deterministic (fixed seeds), fast (seconds, not minutes), and
  must not hit the network.
- Keep tests OS-agnostic: use `pathlib` / `tmp_path`, not hard-coded `/` paths
  (CI runs on Windows).

## Changes, commits and PRs

- Small, focused PRs. One concern per PR.
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change.
- Reference the backlog item or issue (e.g. `BL-013`, `#44`) in the PR body.
- PR description must state: what changed, why, how it was tested, and whether
  any numeric output changed.
- Do not commit notebook outputs outside `case_studies/` (pre-commit runs
  `nbstripout`).

## For automated agents specifically

- Read this file and `CONTRIBUTING.md` before acting.
- If you are not sure a change is safe, open an issue or leave a review comment
  instead of opening a PR.
- Never push to `main`. Never merge. Humans merge.
- Disclose that your PR or comment was AI-generated.
