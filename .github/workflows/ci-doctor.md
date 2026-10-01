---
description: |
  When the main CI workflow fails, investigates the logs, finds the root cause
  and posts a diagnosis with a concrete fix — on the pull request if there is
  one, otherwise as an issue for failures on main.

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

if: ${{ github.event.workflow_run.conclusion == 'failure' }}

permissions:
  actions: read
  contents: read
  issues: read
  pull-requests: read

engine: claude

tools:
  github:
    toolsets: [actions, pull_requests, issues, repos]

safe-outputs:
  add-comment:
    max: 1
    target: "*"
    hide-older-comments: true
  create-issue:
    title-prefix: "[CI failure] "
    labels: [ci, automation]
    max: 1

timeout-minutes: 15
---

# fairpipe CI doctor

The `CI` workflow failed.

- Run: ${{ github.event.workflow_run.html_url }}
- Run ID: ${{ github.event.workflow_run.id }}
- Head SHA: ${{ github.event.workflow_run.head_sha }}
- Event: ${{ github.event.workflow_run.event }}

Read `AGENTS.md` first — it lists the exact commands CI runs.

## Investigate

1. List the run's jobs. CI is a matrix of ubuntu/macos/windows ×
   Python 3.10/3.11/3.12. Note **which** combinations failed: one OS only
   usually means a path, line-ending or platform issue; one Python version
   usually means a dependency or syntax issue; everything failing usually means
   the change itself.
2. Fetch logs for the earliest failed step. Find the first real error, not the
   cascade after it. fairpipe's CI steps are, in order: install → black →
   flake8 → pytest + `coverage report --fail-under=85` → fairness smoke gate →
   CLI report → benchmarks (non-blocking).
3. Classify: lint/format · test failure · coverage dropped below 85% ·
   fairness gate · dependency/install · runner/network/flaky.
4. Look at the commit(s) in the head SHA and connect the failure to the exact
   lines that caused it.
5. Search open issues titled `[CI failure]` for the same error so you don't
   duplicate.

Common fairpipe-specific causes worth checking:

- black version drift (CI pins `black==25.9.0`).
- coverage dropped because new code has no tests.
- `--maxfail=1` in pytest `addopts` means only the first failing test shows;
  say so if more failures are likely hidden.
- Windows-only failures from hard-coded `/` paths or file locking.
- an optional dependency imported at module top level.

## Report

Write a comment with:

- **What failed** — job(s), step, and the key error line (short quote).
- **Root cause** — with confidence (high / medium / low) and the evidence.
- **Fix** — the exact command or code change. For black/flake8, give the
  command to run locally (`black fairness_pipeline_dev_toolkit tests`).
- **Prevent** — one line, only if there is a real prevention (e.g. a missing
  pre-commit hook).
- End with `_Diagnosis by an AI agent (fairpipe ci-doctor workflow)._`

Where to post:

- If the run is associated with an open pull request, use `add-comment` on
  that pull request.
- If it is a push to `main` (no PR), and no open `[CI failure]` issue already
  covers this error, use `create-issue`. If one does, comment on it instead.
- Otherwise (e.g. a push to a feature branch with no PR), do nothing.

Do not propose unrelated cleanup. Do not open pull requests.
