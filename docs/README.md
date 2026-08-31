# Documentation Site

This directory contains the source files for the documentation site.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Or install from the project root:

```bash
pip install -e .[dev]
pip install sphinx sphinx-rtd-theme myst-parser
```

### Build Commands

Build HTML documentation:

```bash
cd docs
make html
```

Or using Sphinx directly:

```bash
cd docs
sphinx-build -b html . _build/html
```

The built documentation will be in `docs/_build/html/`.

### View Locally

After building, open `docs/_build/html/index.html` in your browser.

## Documentation Structure

- `index.rst`: Main documentation index
- `conf.py`: Sphinx configuration
- `getting_started.md`: Getting started guide
- `llm_evals_intro.md`: LLM fairness evals (Phase 0–2)
- `*.md`: Additional documentation files (converted to HTML by MyST parser)

## Continuous Integration

The documentation is automatically built and deployed to GitHub Pages via the `.github/workflows/docs.yml` workflow when changes are pushed to the `main` branch.

### Fixing a 404 at the docs URL

If **https://svrusio.github.io/fAIr** returns 404, GitHub Pages is likely not using the workflow:

1. On GitHub, open the repository **SvrusIO/fAIr**.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions** (not "Deploy from a branch").
4. Save. The next successful run of the **Documentation** workflow (on push to `main` that touches `docs/`, `*.md`, or package code) will deploy the site.
5. To deploy immediately without a code change: **Actions → Documentation → Run workflow** (the workflow has `workflow_dispatch`).

The site will be available at **https://svrusio.github.io/fAIr** (or **https://SvrusIO.github.io/fAIr**).

## Adding New Documentation

1. Add Markdown files to the `docs/` directory
2. Update `index.rst` to include the new file in the table of contents
3. Commit and push - the documentation will be automatically built and deployed

## ReadTheDocs Alternative

To use ReadTheDocs instead of GitHub Pages:

1. Create a `readthedocs.yml` file in the project root
2. Configure ReadTheDocs to build from the `docs/` directory
3. Update the documentation URL in `pyproject.toml`

Example `readthedocs.yml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```
