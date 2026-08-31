"""Sample production LLM logs into the existing tracker / drift stack.

Production records are already-produced completions. This module never calls a
provider: it keeps 1/N rows, reduces each kept row to a group label plus a 0/1
score, and maps that onto ``RealTimeFairnessTracker.process_batch``. The
statistic is unpaired group-rate disparity (max−min of group means) — the same
shape as ``scoring.rate_disparity`` and the tracker's ``_demographic_parity``.
Matched-pairing (``iter_matched_pairs``) is counterfactual-only and does not
apply to untemplated production logs.

Transcript / prompt columns are dropped before ingest so they never land in
``window_df`` or ``metrics_ts``.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE
from fairness_pipeline_dev_toolkit.monitoring.tracker import (
    ColumnMap,
    RealTimeFairnessTracker,
    TrackerConfig,
)

# Columns that must not reach the tracker buffer or default reports.
_TRANSCRIPT_COLUMNS = frozenset(
    {
        "prompt",
        "response",
        "text",
        "completion",
        "output",
        "transcript",
        "messages",
        "content",
    }
)

Records = Union[pd.DataFrame, Sequence[Mapping[str, Any]]]
Scorer = Callable[[str], float]

# Mix into the omitted-seed path so successive production calls do not reuse one draw.
_SAMPLE_CALL_COUNTER = itertools.count()
_GOLDEN_RATIO_ODD = 0x9E3779B97F4A7C15


def _resolve_random_state(random_state: Optional[int]) -> int:
    """Caller seed, or timestamp ⊕ call-counter when ``random_state`` is omitted."""
    if random_state is not None:
        return int(random_state)
    nonce = next(_SAMPLE_CALL_COUNTER)
    return int((time.time_ns() ^ (nonce * _GOLDEN_RATIO_ODD)) & 0xFFFFFFFFFFFFFFFF)


def _as_frame(records: Records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    return pd.DataFrame(list(records))


def _drop_transcripts(df: pd.DataFrame, extra: Iterable[str] = ()) -> pd.DataFrame:
    drop = [col for col in set(_TRANSCRIPT_COLUMNS).union(extra) if col in df.columns]
    return df.drop(columns=drop) if drop else df


def _keep_indices(n_rows: int, sample_every: int, random_state: Optional[int]) -> np.ndarray:
    """Keep exactly ``n_rows // sample_every`` rows (all rows when ``sample_every==1``).

    Chosen positions are returned in increasing order so kept rows stay in their
    original relative (stream) order — required for the windowed tracker's
    time-based drift path.
    """
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1 (1/N), got {sample_every}")
    if n_rows == 0:
        return np.array([], dtype=int)
    if sample_every == 1:
        return np.arange(n_rows, dtype=int)
    n_keep = n_rows // sample_every
    if n_keep == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(_resolve_random_state(random_state))
    chosen = rng.choice(n_rows, size=n_keep, replace=False)
    return np.sort(chosen)


def _binarize_scores(values: Any, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return (arr >= threshold).astype(int)


def sample_production_llm_records(
    records: Records,
    *,
    sample_every: int,
    group_col: str,
    random_state: Optional[int] = None,
    score_col: Optional[str] = None,
    text_col: Optional[str] = None,
    scorer: Optional[Scorer] = None,
    y_true_col: Optional[str] = None,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """Keep 1/``sample_every`` production rows as ``group`` + 0/1 ``y_pred``.

    Sampling is without replacement. The kept count is exactly
    ``len(records) // sample_every``. ``sample_every=1`` keeps every row, in
    input order, without drawing from the RNG.

    ``random_state`` is caller-supplied so successive production batches can
    vary the draw (batch id, timestamp, or a counter). Tests pin an explicit
    int. When omitted, the draw is timestamp ⊕ a process-local call counter —
    it is **not** a hardcoded seed. Kept rows stay in original relative order
    (positions sorted after the draw) so a sliding-window tracker still sees
    time in log order.

    Provide either a precomputed ``score_col`` or a ``text_col`` plus ``scorer``.
    Scores are binarized at ``score_threshold``. Transcript columns are dropped
    from the result. The original row index is preserved so callers can tell
    which source rows were kept.

    This function does not make provider HTTP calls.
    """
    frame = _as_frame(records)
    if group_col not in frame.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if score_col is None and (text_col is None or scorer is None):
        raise ValueError("Provide score_col, or text_col and scorer, to produce a 0/1 score")
    if score_col is not None and score_col not in frame.columns:
        raise KeyError(f"Missing score column: {score_col}")
    if text_col is not None and scorer is not None and text_col not in frame.columns:
        raise KeyError(f"Missing text column: {text_col}")
    if y_true_col is not None and y_true_col not in frame.columns:
        raise KeyError(f"Missing y_true column: {y_true_col}")

    keep = _keep_indices(len(frame), sample_every, random_state)
    kept = frame.iloc[keep]
    if score_col is not None:
        raw_scores = kept[score_col]
    else:
        raw_scores = kept[text_col].map(lambda value: scorer(value or ""))

    out = pd.DataFrame({group_col: kept[group_col].to_numpy()}, index=kept.index)
    out["y_pred"] = _binarize_scores(raw_scores, threshold=score_threshold)
    if y_true_col is not None:
        out[y_true_col] = kept[y_true_col].to_numpy()
    extra_drop = [text_col] if text_col else []
    return _drop_transcripts(out, extra=extra_drop)


def production_llm_column_map(
    *,
    group_col: str = "group",
    score_col: str = "y_pred",
    y_true_col: Optional[str] = None,
) -> ColumnMap:
    """``ColumnMap`` for sampled production LLM rows (``protected=``, not ``sensitive=``)."""
    return ColumnMap(
        y_pred=score_col,
        y_true=y_true_col,
        protected=(group_col,),
    )


def make_production_llm_tracker(
    *,
    window_size: int = 10_000,
    min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE,
    y_true: bool = False,
    metrics: Optional[Sequence[str]] = None,
    artifacts_dir: str = "artifacts/monitoring",
) -> RealTimeFairnessTracker:
    """Tracker configured for production LLM logs.

    Score-only (no gold labels) is the default: ``metrics=("demographic_parity",)``.
    Equalized odds is included only when ``y_true=True`` or when ``metrics`` is
    passed explicitly. ``min_group_size`` defaults to
    ``DEFAULT_LLM_MIN_GROUP_SIZE`` (5), not the classifier tracker’s 30.
    """
    if metrics is None:
        metrics = ("demographic_parity", "equalized_odds") if y_true else ("demographic_parity",)
    cfg = TrackerConfig(
        window_size=window_size,
        min_group_size=min_group_size,
        metrics=tuple(metrics),
    )
    return RealTimeFairnessTracker(cfg, artifacts_dir=artifacts_dir)


def ingest_sampled_production_llm(
    tracker: RealTimeFairnessTracker,
    sampled: pd.DataFrame,
    *,
    group_col: str = "group",
    score_col: str = "y_pred",
    y_true_col: Optional[str] = None,
) -> pd.DataFrame:
    """``process_batch`` a sampled LLM frame; strips leftover transcript columns."""
    batch = _drop_transcripts(sampled)
    cmap = production_llm_column_map(
        group_col=group_col,
        score_col=score_col,
        y_true_col=y_true_col,
    )
    return tracker.process_batch(batch, cmap)
