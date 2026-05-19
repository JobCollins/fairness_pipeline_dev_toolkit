# fairpipe

**Fairness measurement, mitigation, monitoring, and pipeline tooling** for ML workflows.  
PyPI package: **[fairpipe](https://pypi.org/project/fairpipe/)** · License: **Apache-2.0** · Python **3.10+**

| | Fairlearn | AIF360 | **fairpipe** |
|---|---|---|---|
| Metrics library | ✅ | ✅ | ✅ |
| End-to-end pipeline | ❌ | ❌ | ✅ |
| CI/CD integration | ❌ | ❌ | ✅ |
| Production monitoring | ❌ | ❌ | ✅ |
| GitHub Action | ❌ | ❌ | ✅ |
| REST API | ❌ | ❌ | ✅ |
| DataFrame + Parquet I/O | ❌ | ❌ | ✅ |

[![PyPI version](https://img.shields.io/pypi/v/fairpipe.svg)](https://pypi.org/project/fairpipe/)
[![Python versions](https://img.shields.io/pypi/pyversions/fairpipe.svg)](https://pypi.org/project/fairpipe/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-green)](https://github.com/SvrusIO/fAIr)
[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SvrusIO/fAIr/main?urlpath=%2Fdoc%2Ftree%2Fcase_studies%2Fcompas_racial_bias.ipynb)

---

## Install

```bash
pip install fairpipe
```

**Optional extras:** `pip install 'fairpipe[api]'` · `'fairpipe[training]'` · `'fairpipe[monitoring]'` · `'fairpipe[adapters]'`  
(REST API, PyTorch training helpers, dashboards/drift, Fairlearn/Aequitas backends.) Full detail is in the **documentation** below—not duplicated here.

---

## Documentation

**Start here (hosted):** **[Documentation — SvrusIO.github.io/fAIr](https://SvrusIO.github.io/fAIr)**  
Built from this repo’s Sphinx sources; includes getting started, user guide, API reference, integration, performance, and security links.

**In-repo references** (for browsing on GitHub or a checkout):

| Topic | Location |
|--------|----------|
| Getting started | [docs/getting_started.md](docs/getting_started.md) |
| User guide (long-form) | [DOCS.md](DOCS.md) |
| API reference | [docs/api.md](docs/api.md) |
| Playbook · fairpipe (as implemented) | [docs/playbook-part-five-fairpipe.md](docs/playbook-part-five-fairpipe.md) |
| Integration guide | [docs/integration_guide.md](docs/integration_guide.md) |
| Architecture / ADR | [docs/ADR-001-architecture.md](docs/ADR-001-architecture.md) |
| Versioning | [docs/VERSIONING.md](docs/VERSIONING.md) |

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

**[COMPAS Recidivism Bias Analysis](case_studies/compas_racial_bias.ipynb)** —
Reproduces ProPublica's 2016 Machine Bias finding. Measures an Equalized Odds
Difference of **0.2116** on the COMPAS dataset and demonstrates a **53.9% reduction**
via Instance Reweighting.

[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SvrusIO/fAIr/main?filepath=case_studies/compas_racial_bias.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SvrusIO/fAIr/blob/main/case_studies/compas_racial_bias.ipynb)

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
