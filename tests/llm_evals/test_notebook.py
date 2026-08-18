"""Execute the LLM counterfactual fairness case study notebook without live network calls."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_llm_counterfactual_notebook_executes():
    notebook_path = (
        Path(__file__).resolve().parents[2] / "case_studies" / "llm_counterfactual_fairness.ipynb"
    )
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()

    rendered = " ".join(
        out.get("text", "") if isinstance(out.get("text"), str) else "".join(out.get("text", []))
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.outputs
        if out.get("output_type") == "stream"
    )
    assert "Guard correctly blocked below-threshold fixture" in rendered
    assert "ILLUSTRATIVE RUN (allow_small_samples=True" in rendered
    assert "Counterfactual fairness divergence:" in rendered
    assert "Notebook smoke-test check passed" in rendered
