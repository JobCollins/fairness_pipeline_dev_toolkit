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
- `*.md`: Additional documentation files (converted to HTML by MyST parser)

## Continuous Integration

The documentation is automatically built and deployed to GitHub Pages via the `.github/workflows/docs.yml` workflow when changes are pushed to the `main` branch.

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
