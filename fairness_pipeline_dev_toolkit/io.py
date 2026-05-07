"""I/O utilities for loading tabular data from various file formats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a tabular data file into a DataFrame.

    Supports .csv, .parquet, and .pq files. Format is detected automatically
    from the file extension — no ``--format`` flag needed.

    Args:
        path: Path to the data file.

    Returns:
        A ``pd.DataFrame`` containing the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .csv, .parquet, .pq")
