from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

def build_intersectional_labels(
        attrs_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sep: str = "||",
        include_na: bool = True
) -> pd.Series:
    """
    Create an intersectional group label per row by concatenating selected attribute columns.
    
    Args:
        attrs_df (pd.DataFrame): Must align 1:1 with y_true/y_pred index/length.
        columns (Optional[List[str]]): List of column names to use for intersectional labels.
                                       If None, all columns in attrs_df are used.
        sep (str): Separator used to join attribute values in the label (avoid characters present in values).
        include_na (bool): Whether to include rows with NaN values in any of the selected columns.
                           If False, rows with NaN values will be excluded from the output.
                           If True, NaN values will be represented as the string "NaN" in the labels.

    Returns:
        pd.Series: A Series of dtype 'category' representing intersectional group labels.
    """
    if columns is None:
        columns = list(attrs_df.columns)

    df = attrs_df[columns].copy()
    
    if include_na:
        # Replace NaN with string "NaN" for labeling
        df = df.fillna("NaN")
        labels = df.astype(str).agg(sep.join, axis=1)
    else:
        # will introduce NaNs where any attribute is NaN
        labels = df.astype("string").agg(sep.join, axis=1)

    return labels.astype('category')

def min_group_mask(
        labels: Iterable,
        min_group_size: int
) -> np.ndarray:
    """
    Return a boolean mask slecting rows whose group appears >= min_group_size times."""
    s = pd.Series(labels)

    # ensure categoricals are not compared againsts ints
    if pd.api.types.is_categorical_dtype(s.dtype):
        s = s.astype(object) # or astype("string") but object is fastest/neutral here
    
    group_counts = s.value_counts(dropna=False)
    # valid_groups = s.map(group_counts) >= min_group_size
    # Map to per-row group size and coerce to numeric for safe comparison
    per_row_sizes = s.map(group_counts).astype("Int64")

    return (per_row_sizes >= int(min_group_size)).to_numpy()

def group_sizes(
        labels: Iterable
) -> Dict[str, int]:
    """
    Return a dictionary of group -> size, helpful in reporting n_per_group outputs.
    Only counts non-NaN labels.
    """
    s = pd.Series(labels)

    if pd.api.types.is_categorical_dtype(s.dtype):
        s = s.astype(object)
    
    counts =  s.value_counts(dropna=True)
    return {str(k): int(v) for k, v in counts.to_dict().items()}
