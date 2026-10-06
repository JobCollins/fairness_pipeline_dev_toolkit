"""Execute README Quick start code blocks exactly as published (BL-018)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _extract_fenced_blocks(markdown: str) -> list[tuple[str, str]]:
    """Return (language, body) for each ```lang ... ``` fence."""
    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    return [(m.group(1).strip() or "text", m.group(2)) for m in pattern.finditer(markdown)]


def _quickstart_section(markdown: str) -> str:
    match = re.search(
        r"## Quick start\n(.*?)(?=\n## |\Z)",
        markdown,
        flags=re.DOTALL,
    )
    assert match, "README missing '## Quick start' section"
    return match.group(1)


@pytest.mark.parametrize("lang", ["python"])
def test_readme_quickstart_python_blocks(lang: str, tmp_path: Path) -> None:
    section = _quickstart_section(README.read_text(encoding="utf-8"))
    blocks = [body for block_lang, body in _extract_fenced_blocks(section) if block_lang == lang]
    assert blocks, f"No ```{lang}``` blocks under Quick start"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for i, body in enumerate(blocks):
        script = tmp_path / f"quickstart_{i}.py"
        script.write_text(textwrap.dedent(body), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            f"README Quick start python block {i} failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def test_fairpipe_io_importable() -> None:
    import fairpipe.io as fio
    from fairpipe.io import load_data

    assert callable(load_data)
    assert fio.load_data is load_data


def test_fairpipe_pipeline_config_importable() -> None:
    from fairpipe.pipeline.config import PipelineConfig, find_config_file

    assert PipelineConfig is not None
    assert callable(find_config_file)
