# fairpipe

**Fairness measurement, mitigation, monitoring, and pipeline tooling** for ML workflows.  
PyPI package: **[fairpipe](https://pypi.org/project/fairpipe/)** · License: **Apache-2.0** · Python **3.10+**

[![PyPI version](https://img.shields.io/pypi/v/fairpipe.svg)](https://pypi.org/project/fairpipe/)
[![Python versions](https://img.shields.io/pypi/pyversions/fairpipe.svg)](https://pypi.org/project/fairpipe/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-green)](https://github.com/SvrusIO/fAIr)

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

## Development

```bash
git clone https://github.com/SvrusIO/fAIr.git
cd fAIr
pip install -e ".[dev]"
pytest -q
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** and **[SECURITY.md](SECURITY.md)**.

---

## Optional: GitHub Action for CI

Example composite action (metrics + optional threshold gate): **[SvrusIO/fairpipe-action](https://github.com/SvrusIO/fairpipe-action)** — usage snippets also appear in the integration / CI sections of the **hosted documentation**.

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
