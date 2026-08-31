# fairpipe

**Fairness measurement, mitigation, monitoring, and pipeline tooling** for ML workflows.  
PyPI package: **[fairpipe](https://pypi.org/project/fairpipe/)** · License: **Apache-2.0** · Python **3.10+**

| | Fairlearn | AIF360 | **fairpipe** |
|---|---|---|---|
| Metrics library | ✅ | ✅ | ✅ |
| Mitigation algorithms | ✅ | ✅ | ✅ |
| DataFrame I/O | ✅ | ✅ | ✅ |
| Parquet I/O | ❌ | ❌ | ✅ |
| Orchestrated end-to-end pipeline | ⚠️ Partial | ⚠️ Partial | ✅ |
| CI/CD integration | ❌ | ❌ | ✅ |
| GitHub Action | ❌ | ❌ | ✅ |
| Production monitoring | ❌ | ❌ | ✅ |
| REST API | ❌ | ❌ | ✅ |
| LLM / GenAI fairness evals | ❌ | ❌ | ✅ |

`Fairlearn` and `AIF360` provide individual pre/in/post-processing components; `fairpipe` provides a YAML-configured `baseline→transform→validate` workflow with CI/CD exit codes.

[![PyPI version](https://img.shields.io/pypi/v/fairpipe.svg)](https://pypi.org/project/fairpipe/)
[![Python versions](https://img.shields.io/pypi/pyversions/fairpipe.svg)](https://pypi.org/project/fairpipe/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-green)](https://github.com/SvrusIO/fAIr)
[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SvrusIO/fAIr/main?urlpath=%2Fdoc%2Ftree%2Fcase_studies%2Fcompas_racial_bias.ipynb)

---

## Install

```bash
pip install fairpipe
```

**Optional extras:** `pip install 'fairpipe[api]'` · `'fairpipe[training]'` · `'fairpipe[monitoring]'` · `'fairpipe[adapters]'` · `'fairpipe[llm]'`  
(REST API, PyTorch training helpers, dashboards/drift, Fairlearn/Aequitas backends, LLM provider SDKs.) Full detail is in the **documentation** below—not duplicated here.

---

## Setting LLM provider credentials

LLM fairness evaluation (see `fairpipe.llm_evals`) uses provider SDKs installed via the optional extra:

```bash
pip install 'fairpipe[llm]'
```

**Credentials are read from environment variables only** — never from YAML config files or CLI flags:

| Provider | Environment variable | Notes |
|----------|---------------------|--------|
| OpenAI (and OpenAI-compatible APIs) | `OPENAI_API_KEY` | Required for `provider: openai` |
| Anthropic | `ANTHROPIC_API_KEY` | Required for `provider: anthropic` |
| Local / self-hosted | *(none)* | `provider: local` needs no API key |

Toxicity/sentiment disparity uses a **lexical scorer by default** (no moderation API key). Plug in
an external scorer via `ToxicitySentimentEvaluator.run_async(scorer=...)`.

Example:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="..."
```

When using the CLI, run `fairpipe llm-eval --dry-run` to estimate request volume and approximate cost before making live provider calls. Live HTTP is **forbidden by default**; set `FAIRPIPE_LLM_ALLOW_LIVE=1` on CLI, REST, Jupyter, or CI jobs that should call a provider (same flag — see [Environment Variables](docs/integration_guide.md#environment-variables)). Replay-from-`cache_dir` does not need it.

```bash
fairpipe llm-eval --config llm_eval.yml --dry-run
fairpipe llm-eval --config llm_eval.yml --report-md artifacts/llm_report.md --with-ci
fairpipe llm-eval --config llm_eval.yml --metric counterfactual_fairness_divergence --threshold 0.25
```

See **[docs/llm_evals_intro.md](docs/llm_evals_intro.md)** for configuration, REST `POST /llm-eval`, and sampling production logs into the existing tracker.

---

## Documentation

**Start here (hosted):** **[Documentation — SvrusIO.github.io/fAIr](https://SvrusIO.github.io/fAIr)**  
Built from this repo’s Sphinx sources; includes getting started, user guide, API reference, integration, performance, and security links.

**In-repo references** (for browsing on GitHub or a checkout):

| Topic | Location |
|--------|----------|
| LLM fairness evals | [docs/llm_evals_intro.md](docs/llm_evals_intro.md) |
| Getting started | [docs/getting_started.md](docs/getting_started.md) |
| User guide (long-form) | [DOCS.md](DOCS.md) |
| API reference | [docs/api.md](docs/api.md) |
| Playbook · fairpipe (as implemented) | [docs/playbook-part-five-fairpipe.md](docs/playbook-part-five-fairpipe.md) |
| Integration guide | [docs/integration_guide.md](docs/integration_guide.md) |
| Architecture / ADR | [docs/ADR-001-architecture.md](docs/ADR-001-architecture.md) |
| Versioning | [docs/VERSIONING.md](docs/VERSIONING.md) |
| Release checklist (mirror / PyPI) | [docs/RELEASE.md](docs/RELEASE.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

---

## Quick start

**CLI**

```bash
fairpipe validate \
  --csv data.csv \
  --y-true y_true \
  --y-pred y_pred \
  --sensitive gender \
  --with-ci

fairpipe run-pipeline --config config.yml --csv data.csv --output-dir artifacts/
```

**Python**

```python
from fairpipe import load_data
from fairpipe.metrics import FairnessAnalyzer

df = load_data("data.csv")
analyzer = FairnessAnalyzer(min_group_size=30)
result = analyzer.demographic_parity_difference(
    y_pred=df["y_pred"],
    sensitive=df["gender"],
    with_ci=True,
)
print(result.value, result.ci)
```

CLI commands, YAML configuration, workflow orchestration, training, monitoring, and the optional REST API are documented on **[the docs site](https://SvrusIO.github.io/fAIr)** and in **[docs/api.md](docs/api.md)**.

---

## CI/CD Integration

Add fairness validation to every pull request with the companion GitHub Action:

```yaml
# .github/workflows/fairness-check.yml
name: Fairness Check
on: [pull_request]

jobs:
  fairness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: SvrusIO/fairpipe-action@v1
        with:
          csv: data/predictions.csv
          y-true: y_true
          y-pred: y_pred
          sensitive: gender
          threshold: "0.05"
          metric: "equalized_odds_difference"
          fail-on-violation: "true"
```

Point `csv` at your predictions file. If equalized odds difference exceeds `0.05`, the PR is blocked. A full fairness report is written to the Actions job summary — metric values, confidence intervals, group breakdowns — permanently attached to the commit.

Gate LLM fairness evals the same way. Live provider HTTP is **forbidden by default** (`LiveLLMCallForbidden`); a job that should call a provider must set `FAIRPIPE_LLM_ALLOW_LIVE=1` — that is the correct safe default, not a workaround. Replay-from-`cache_dir` jobs do not need the flag.

```yaml
# .github/workflows/llm-fairness-check.yml
name: LLM Fairness Check
on: [pull_request]

jobs:
  llm-fairness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: SvrusIO/fairpipe-action@v1
        env:
          FAIRPIPE_LLM_ALLOW_LIVE: "1"
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        with:
          config: llm_eval.yml
          metric: "counterfactual_fairness_divergence"
          threshold: "0.25"
          fail-on-violation: "true"
```

A red check can be decoded without opening the report:

| Exit | `gate_status` | Meaning |
|------|----------------|---------|
| 0 | `pass` | Threshold met (or no threshold) on a non-caveated metric |
| 1 | `fail` | Threshold miss on a **non-caveated** gated metric |
| 2 | *(usage)* | `--threshold` without `--metric`, unknown metric, cache miss / live-forbidden |
| 3 | `illustrative` | Gated metric has a non-null `caveat` — **even if the number would pass** |

`llm-fairness-check` mode in the Action is a companion-repo follow-up ([BL-010](docs/fairpipe-technical-backlog.md)). This package already exposes the same `with:` keys via `fairpipe llm-eval --threshold` / `--metric` and `run_llm_fairness_check()`.

→ **[SvrusIO/fairpipe-action](https://github.com/SvrusIO/fairpipe-action)**

---

## Development

```bash
git clone https://github.com/SvrusIO/fAIr.git
cd fAIr
pip install -e ".[dev]"
pytest -q
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** and **[SECURITY.md](SECURITY.md)**.

---

## Case Studies

Real-world bias audits demonstrating fairpipe's full pipeline — from
measurement and detection through mitigation and CI/CD integration.

### [LLM Counterfactual Fairness](case_studies/llm_counterfactual_fairness.ipynb)

Measures gender-coded divergence in LLM hiring recommendations using the counterfactual
fairness probe with **committed live-recorded Anthropic responses** replayed from cache
(no API key required). Select kernel **Python (fairpipe .venv)** if imports fail.

- **Part A** — n=1 per group → **`nan`** at default `min_group_size=5` (guard demonstration)
- **Part B** — n=9 per group → divergence **≈ 0.196** (95% CI ≈ 0.185–0.205) on lexical
  features; this is **not** “19.6% of candidates treated unfairly”
- YAML config → `run_llm_eval()` → `MetricResult` (see `docs/llm_evals_intro.md`). Phase 2
  refusal/toxicity/BBQ demo caches are labeled via `MetricResult.caveat` until BL-009;
  they are **not** part of this notebook.

---

### [COMPAS Recidivism Bias Analysis](https://github.com/SvrusIO/fAIr/blob/main/case_studies/compas_racial_bias.ipynb)

Reproduces ProPublica's 2016 *Machine Bias* investigation on the COMPAS
recidivism algorithm used in US courtrooms.

- **DPD = 0.2451** — Black defendants 24.5 percentage points more likely
  to be flagged high-risk than white defendants
- **EOD = 0.2116** — among defendants who will not reoffend, Black
  defendants are 21 percentage points more likely to be incorrectly
  labelled high-risk
- **53.9% reduction in EOD** via Instance Reweighting
- 28 features with statistically significant racial disparities detected
- 23 proxy variables identified — removing the race column alone would
  not fix this model

[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SvrusIO/fAIr/main?filepath=case_studies/compas_racial_bias.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SvrusIO/fAIr/blob/main/case_studies/compas_racial_bias.ipynb)


### [AI Hiring Bias — ACS Employment Analysis](https://github.com/SvrusIO/fAIr/blob/main/case_studies/acs_employment_bias.ipynb)

Demonstrates the type of bias audit now required under NYC Local Law 144
and the EU AI Act, framed around *Mobley v. Workday* — the 2025 class
action alleging AI hiring tools discriminated against millions of
applicants by age, race, and disability.

- **DPD = 0.1046** — white candidates selected at 32.3% vs Black
  candidates at 21.8%, a 10.5 percentage point gap with no race feature
  in the model
- **EOD = 0.1022** — 76.8% of qualified Black candidates incorrectly
  rejected vs 66.6% of white candidates
- All 5 prediction features show statistically significant racial
  disparity — removing the race column alone would not fix this model
- **47.6% reduction in EOD** via Instance Reweighting, closing to within
  0.0036 of the 0.05 compliance threshold
- Dataset: ACS 2018 1-Year California (196,604 individuals, folktables)

[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SvrusIO/fAIr/main?urlpath=%2Fdoc%2Ftree%2Fcase_studies%2Facs_employment.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SvrusIO/fAIr/blob/main/case_studies/acs_employment.ipynb)


---

## Project links

| | |
|--|--|
| **Homepage / docs** | [SvrusIO.github.io/fAIr](https://SvrusIO.github.io/fAIr) |
| **Repository** | [github.com/SvrusIO/fAIr](https://github.com/SvrusIO/fAIr) |
| **Issues** | [github.com/SvrusIO/fAIr/issues](https://github.com/SvrusIO/fAIr/issues) |

---

## License

Apache License 2.0 — see **[LICENSE](LICENSE)**.
