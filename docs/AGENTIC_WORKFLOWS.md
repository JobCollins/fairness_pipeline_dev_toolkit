# Agentic workflows and quality gates

This repo uses two layers to stop broken code reaching `main`.

| Layer | What it is | Blocks a merge? |
|---|---|---|
| **Deterministic gates** | pre-commit hooks locally, `ci.yml` in Actions, branch protection on `main` | **Yes** — this is the real gate |
| **AI agents** ([GitHub Agentic Workflows](https://github.github.com/gh-aw/)) | Review PRs, diagnose CI failures, propose performance PRs | No — advisory; a human decides |

Agents are good at things linters and tests can't see: a metric that is now
silently wrong, a public API that was renamed without a deprecation, a missing
regression test. They are not a replacement for tests. Never make an AI review
the only thing between a commit and `main`.

## What's here

| File | Role |
|---|---|
| `AGENTS.md` | Standing instructions every coding agent reads (Copilot, Claude Code, Cursor, Codex, gh-aw). The single source of truth for commands and rules. |
| `.github/workflows/pr-review.md` | Reviews every non-draft PR. Inline comments + one summary review. Can run code to verify a suspected bug before reporting it. |
| `.github/workflows/ci-doctor.md` | Runs when `CI` fails. Reads the logs and posts root cause + fix on the PR (or opens an issue for `main`). |
| `.github/workflows/perf-improver.md` | Weekly (or manual). Benchmarks, makes one optimisation, proves metric values are unchanged, opens a **draft** PR. At most one open at a time. |
| `*.lock.yml` | Compiled GitHub Actions generated from the `.md` files. **Never edit by hand**; edit the `.md` and recompile. |
| `.pre-commit-config.yaml` | black (same version as CI), isort, ruff, flake8, nbstripout on commit; fast pytest on push. |

## One-time setup

### 1. Local hooks (stops bad commits on your machine)

```bash
pip install -e ".[dev]"
pre-commit install                       # runs on every commit
pre-commit install --hook-type pre-push  # runs pytest before every push
pre-commit run --all-files               # check the whole repo once
```

If the full test suite is too slow for every push, narrow the `entry` of the
`pytest-before-push` hook (for example `python -m pytest -q -x tests/metrics tests/stats`).

### 2. Protect `main` (stops bad merges on GitHub)

GitHub → **Settings → Rules → Rulesets → New branch ruleset** (or *Branches →
Add rule*) for `main`:

- ✅ Require a pull request before merging
- ✅ Require status checks to pass → add the `CI` jobs, e.g.
  `test (ubuntu-latest, 3.12)`, `test (windows-latest, 3.10)`, … (they appear
  in the picker after CI has run once on a PR)
- ✅ Block force pushes
- Optional: require conversation resolution, so agent review comments must be
  answered before merge

From then on, all work goes through a branch and a PR, even when you work alone.

### 3. Install the gh-aw CLI

```bash
gh extension install github/gh-aw
gh aw version
```

### 4. Add the AI engine secret

The workflows use `engine: claude`, so they need an Anthropic API key:

```bash
gh secret set ANTHROPIC_API_KEY        # paste the key when prompted
```

To use Copilot instead, change `engine: claude` to `engine: copilot` in each
`.md`, set `COPILOT_GITHUB_TOKEN` (a fine-grained PAT with "Copilot Requests"
permission), and recompile. Codex: `engine: codex` + `OPENAI_API_KEY`.

### 5. Compile and commit

```bash
gh aw compile          # regenerates .github/workflows/*.lock.yml
git add AGENTS.md .github/ .gitattributes .pre-commit-config.yaml docs/AGENTIC_WORKFLOWS.md
git commit -m "Add AGENTS.md, agentic workflows, and aligned pre-commit gates"
```

Push on a branch, open a PR, and the `pr-review` workflow will review its own
introduction — a good first test.

### 6. Try the others

```bash
gh aw run perf-improver               # manual run (or Actions tab → Run workflow)
gh aw logs pr-review                  # download and inspect recent agent runs
gh aw audit <run-id>                  # deep-dive one run: tools used, cost, errors
```

`ci-doctor` triggers itself the next time `CI` fails.

## Day-to-day

- **Change an agent's behaviour:** edit the Markdown body of its `.md`. Changes
  to the *frontmatter* (between the `---` lines) require `gh aw compile`.
- **Change project rules:** edit `AGENTS.md`. All agents pick it up, including
  your local Cursor / Claude Code sessions.
- **Too noisy?** Lower `max:` under `safe-outputs`, or limit `pr-review` to
  certain paths by adding `paths: ["fairness_pipeline_dev_toolkit/**"]` under
  `on.pull_request` and recompiling.
- **Cost:** `pr-review` runs on every push to a non-draft PR. Keep PRs in
  draft while iterating, and mark them ready when you want the review. Dependabot
  PRs are skipped.

## Security model, briefly

- The agent job runs with **read-only** permissions in a sandbox with a network
  firewall (only `defaults` + PyPI allowed here).
- Writes (comments, reviews, PRs, issues) go through **safe outputs**: separate,
  permission-scoped jobs that apply only the action types and limits listed in
  the frontmatter.
- Only users with write access trigger the PR reviewer; forks don't get secrets.
- The agents never push to `main` or merge.

## Mirrors

`pyproject.toml` lists `SvrusIO/fAIr` as the canonical repo. Install the
workflows and the `ANTHROPIC_API_KEY` secret on whichever repo PRs are actually
opened against. The `.md` / `.lock.yml` files work unchanged in either.
