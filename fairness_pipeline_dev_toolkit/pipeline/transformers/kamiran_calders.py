"""Kamiran & Calders (2012) label-aware reweighing."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fairness_pipeline_dev_toolkit.exceptions import KamiranCaldersLabelError

# Joint group key: single attribute value, or tuple for multiple attributes.
GroupKey = Union[Hashable, Tuple[Hashable, ...]]
CellKey = Tuple[GroupKey, int]


def _as_1d_labels(y: Any, *, n_rows: int) -> np.ndarray:
    """Coerce ``y`` to a length-``n_rows`` 1-d array."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise KamiranCaldersLabelError(
                "KamiranCaldersReweighing expects a 1-d label vector; "
                f"got DataFrame with shape {y.shape}.",
                suggestion="Pass a Series/array, or set label= to a single column name in X.",
            )
        arr = y.iloc[:, 0].to_numpy()
    elif isinstance(y, pd.Series):
        arr = y.to_numpy()
    else:
        arr = np.asarray(y)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise KamiranCaldersLabelError(
                f"KamiranCaldersReweighing expects a 1-d label vector; got shape {arr.shape}.",
                suggestion="Pass a 1-d array/Series of binary labels in {0, 1}.",
            )
    if len(arr) != n_rows:
        raise KamiranCaldersLabelError(
            f"Label length {len(arr)} does not match X rows {n_rows}.",
            suggestion="Align y with X (same row order and length) before fit.",
        )
    return arr


def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    """Require binary labels in ``{0, 1}``; reject NaN, multiclass, other encodings."""
    if len(y) == 0:
        return np.asarray(y, dtype=int)

    # Detect NaN / None before numeric coerce loses the distinction for object dtype.
    s = pd.Series(y)
    if s.isna().any():
        n_nan = int(s.isna().sum())
        raise KamiranCaldersLabelError(
            f"KamiranCaldersReweighing requires finite binary labels in {{0, 1}}; "
            f"found {n_nan} NaN/missing label(s).",
            suggestion="Impute or drop rows with missing labels before fitting.",
        )

    if s.dtype == bool or (isinstance(s.dtype, np.dtype) and np.issubdtype(s.dtype, np.bool_)):
        return s.astype(int).to_numpy()

    num = pd.to_numeric(s, errors="coerce")
    if num.isna().any():
        raise KamiranCaldersLabelError(
            "KamiranCaldersReweighing requires binary labels in {0, 1}; "
            "some values could not be interpreted as numeric 0/1.",
            suggestion="Encode labels as integers 0 and 1 (positive class = 1).",
        )

    vals = num.to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise KamiranCaldersLabelError(
            "KamiranCaldersReweighing requires finite binary labels in {0, 1}; "
            "found non-finite values.",
            suggestion="Remove or impute non-finite labels before fitting.",
        )

    uniq = np.unique(vals)
    allowed = {0.0, 1.0}
    if len(uniq) > 2:
        raise KamiranCaldersLabelError(
            f"KamiranCaldersReweighing requires binary labels in {{0, 1}}; "
            f"got {len(uniq)} distinct values {sorted(uniq.tolist())[:8]}"
            f"{'…' if len(uniq) > 8 else ''}.",
            suggestion="Use binary outcomes only, or remap to {0, 1} with positive class 1.",
        )
    if not set(uniq.tolist()) <= allowed:
        raise KamiranCaldersLabelError(
            f"KamiranCaldersReweighing requires labels in {{0, 1}} (positive=1); "
            f"got distinct values {sorted(uniq.tolist())}.",
            suggestion="Remap encodings, e.g. `(y == positive_class).astype(int)`.",
        )
    return vals.astype(int)


class KamiranCaldersReweighing(BaseEstimator, TransformerMixin):
    """Kamiran & Calders (2012) reweighing for binary classification.

    Each training row in joint sensitive group ``s`` with label ``y`` receives:

    .. math::

        w(s, y) = (n_s \\cdot n_y) / (n \\cdot n_{sy})

    That is the count expected if group and label were independent, divided by
    the observed cell count. In the weighted training data, group and label are
    independent, so the weighted positive rate is equal across groups. Without
    clipping, weights sum to exactly ``n`` (mean 1).

    **Guarantees (unclipped):** independence of the (joint) sensitive group and
    the label in the *weighted training sample*. Weighted base rates match.

    **Does not guarantee:** equalized odds; fairness of anything the model
    learns through proxies; stable estimates from small ``(s, y)`` cells.

    Parameters
    ----------
    sensitive :
        Column name(s) for protected attributes. Multiple attributes form a
        **joint** cell (e.g. woman×Black); weights are not multiplied
        per-attribute.
    label :
        Optional column in ``X`` providing binary labels when ``y`` is not
        passed to :meth:`fit`. Prefer ``fit(X, y)`` in training workflows.
    max_weight :
        Optional absolute clip on per-row weights. **Default ``None`` (no
        clipping)** — clipping breaks the exact independence guarantee. When
        set, weights are clipped then renormalized to mean 1; independence
        then holds only approximately.

    Attributes
    ----------
    sample_weight_ :
        Per-row weights aligned to the *fit* frame (train-only artifact).
    cell_weights_ :
        Mapping ``(group, label) -> weight`` actually applied to eligible rows.
    degenerate_groups_ :
        Groups that had only one label value at fit (reweighing cannot
        equalize their base rate); empty if none.
    n_excluded_sensitive_nan_ :
        Rows with NaN in any sensitive column (assigned weight 1; excluded
        from cell counts), matching the package ``nan_policy="exclude"``
        convention for protected attributes.

    Notes
    -----
    Citation: Kamiran, F. & Calders, T. (2012). Data preprocessing techniques
    for classification without discrimination. *Knowledge and Information
    Systems*, 33(1), 1–33.

    Examples
    --------
    >>> import pandas as pd
    >>> from fairpipe.pipeline import KamiranCaldersReweighing
    >>> X = pd.DataFrame({"s": ["A", "A", "B", "B"], "f": [1, 2, 3, 4]})
    >>> y = pd.Series([1, 0, 1, 0])
    >>> kc = KamiranCaldersReweighing(sensitive=["s"]).fit(X, y)
    >>> kc.sample_weight_.sum()  # doctest: +SKIP
    4.0
    """

    def __init__(
        self,
        sensitive: Sequence[str],
        label: Optional[str] = None,
        max_weight: Optional[float] = None,
    ):
        self.sensitive = list(sensitive)
        self.label = label
        self.max_weight = None if max_weight is None else float(max_weight)

        self.sample_weight_: Optional[np.ndarray] = None
        self.cell_weights_: Dict[CellKey, float] = {}
        self.degenerate_groups_: List[GroupKey] = []
        self.n_excluded_sensitive_nan_: int = 0

    def _resolve_labels(self, X: pd.DataFrame, y: Optional[Any]) -> np.ndarray:
        if y is not None:
            return _validate_binary_labels(_as_1d_labels(y, n_rows=len(X)))
        if self.label is not None:
            if self.label not in X.columns:
                raise KamiranCaldersLabelError(
                    f"KamiranCaldersReweighing label column '{self.label}' not found in X.",
                    missing_columns=[self.label],
                    data_shape=X.shape,
                    suggestion=(
                        "Pass y to fit(X, y), or set label= to a column present in X, "
                        "or set training.target_column in YAML so the pipeline defaults label."
                    ),
                )
            return _validate_binary_labels(X[self.label].to_numpy())
        raise KamiranCaldersLabelError(
            "KamiranCaldersReweighing requires binary labels: pass y to fit(X, y), "
            "or set the label= column name so labels are read from X[label].",
            suggestion=(
                "In execute_workflow, y_train is forwarded automatically. "
                "For CLI/REST pipeline runs, set params.label or training.target_column "
                "in the YAML config. Frequency balancing without labels is "
                "InstanceReweighting, not Kamiran–Calders."
            ),
        )

    def _joint_groups(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.sensitive if c not in X.columns]
        if missing:
            raise ValueError(f"Sensitive attribute(s) not found in DataFrame: {missing}")
        if len(self.sensitive) == 1:
            return X[self.sensitive[0]].to_numpy()
        # Joint cell: one hashable tuple per row. Avoid np.array(list_of_tuples),
        # which collapses equal-length tuples into a 2-d array (unhashable rows).
        cols = [X[c].to_numpy() for c in self.sensitive]
        out = np.empty(len(X), dtype=object)
        for i, row in enumerate(zip(*cols)):
            out[i] = tuple(row)
        return out

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "KamiranCaldersReweighing":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("KamiranCaldersReweighing expects a pandas DataFrame X.")
        if not self.sensitive:
            raise ValueError("KamiranCaldersReweighing requires at least one sensitive column.")

        n = len(X)
        if n == 0:
            self.sample_weight_ = np.array([], dtype=float)
            self.cell_weights_ = {}
            self.degenerate_groups_ = []
            self.n_excluded_sensitive_nan_ = 0
            return self

        labels = self._resolve_labels(X, y)
        groups = self._joint_groups(X)

        # Rows with NaN in any sensitive column: weight 1, exclude from counts.
        if len(self.sensitive) == 1:
            sens_nan = pd.isna(groups)
        else:
            sens_nan = np.array([any(pd.isna(v) for v in g) for g in groups], dtype=bool)
        self.n_excluded_sensitive_nan_ = int(sens_nan.sum())

        eligible = ~sens_nan
        w = np.ones(n, dtype=float)
        self.cell_weights_ = {}
        self.degenerate_groups_ = []

        if not eligible.any():
            self.sample_weight_ = w
            return self

        g_el = groups[eligible]
        y_el = labels[eligible]
        n_el = int(eligible.sum())

        # Cell and margin counts on eligible rows only.
        # Use pandas for robust hashing of mixed group keys.
        cell_df = pd.DataFrame({"g": list(g_el), "y": y_el})
        cell_counts = cell_df.groupby(["g", "y"], sort=False).size()
        n_s = cell_df.groupby("g", sort=False).size()
        n_y = cell_df.groupby("y", sort=False).size()

        # Degenerate groups: only one label present.
        labels_per_group = cell_df.groupby("g")["y"].nunique()
        for g, nuniq in labels_per_group.items():
            if int(nuniq) < 2:
                counts = {int(lab): int(cell_counts.get((g, lab), 0)) for lab in (0, 1)}
                warnings.warn(
                    f"KamiranCaldersReweighing: group {g!r} has a single label "
                    f"value (cell counts {counts}); cannot equalize its base rate.",
                    UserWarning,
                    stacklevel=2,
                )
                self.degenerate_groups_.append(g)

        cell_w: Dict[CellKey, float] = {}
        for (g, lab), n_sy in cell_counts.items():
            lab_i = int(lab)
            n_sy_i = int(n_sy)
            if n_sy_i <= 0:
                continue
            weight = (float(n_s[g]) * float(n_y[lab_i])) / (float(n_el) * float(n_sy_i))
            cell_w[(g, lab_i)] = float(weight)

        self.cell_weights_ = cell_w

        # Assign per-row weights for eligible rows.
        for i in np.flatnonzero(eligible):
            key = (groups[i], int(labels[i]))
            w[i] = cell_w.get(key, 1.0)

        if self.max_weight is not None:
            if self.max_weight <= 0:
                raise ValueError("max_weight must be positive when set.")
            lo = 1.0 / self.max_weight
            w = np.clip(w, lo, self.max_weight)
            mu = float(np.mean(w)) if len(w) else 1.0
            if mu > 0:
                w = w / mu

        self.sample_weight_ = w
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sample_weight_ is None:
            raise RuntimeError("KamiranCaldersReweighing must be fitted before transform.")
        # Features unchanged; sample_weight_ stays from fit (train-sized).
        return X
