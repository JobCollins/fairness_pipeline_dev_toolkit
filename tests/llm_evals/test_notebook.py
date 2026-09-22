"""Execute the LLM fairness measurement-pitfalls case study without live network calls."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import nbformat


def test_llm_fairness_measurement_pitfalls_notebook_executes(assert_no_live_llm_calls):
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "case_studies" / "llm_fairness_measurement_pitfalls.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)

    # In-process exec, not nbclient. GitHub CI installs nbclient/nbformat but not
    # ipykernel, so there is no registered ``python3`` kernelspec
    # (NoSuchKernel on macos-latest / Python 3.10). A Jupyter kernel is also a
    # child process, which assert_no_live_llm_calls cannot patch.
    ns: dict = {"__name__": "__main__"}
    captured = io.StringIO()
    previous_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        with redirect_stdout(captured):
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if not str(source).strip():
                    continue
                compiled = compile(source, f"{notebook_path.name}:cell{i}", "exec")
                exec(compiled, ns)  # noqa: S102 — notebook cells are the fixture
    finally:
        os.chdir(previous_cwd)

    rendered = captured.getvalue()
    assert "Guard correctly blocked below-threshold fixture" in rendered
    assert "Part B — expanded fixture at default min_group_size" in rendered
    assert "Demographic swap divergence:" in rendered
    assert "Notebook threshold-clearing check passed." in rendered


def test_legacy_notebook_path_is_stub_only():
    """Old path must remain as a non-executable stub (exclude from live exec)."""
    repo_root = Path(__file__).resolve().parents[2]
    stub = repo_root / "case_studies" / "llm_counterfactual_fairness.ipynb"
    nb = nbformat.read(stub, as_version=4)
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == "markdown"
    assert "llm_fairness_measurement_pitfalls.ipynb" in "".join(nb.cells[0].source)
    assert not any(c.cell_type == "code" for c in nb.cells)
