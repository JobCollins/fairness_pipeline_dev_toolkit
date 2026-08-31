"""Production LLM sampling → existing tracker / drift engine."""

from __future__ import annotations

import pandas as pd

from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE
from fairness_pipeline_dev_toolkit.llm_evals.production import (
    ingest_sampled_production_llm,
    make_production_llm_tracker,
    sample_production_llm_records,
)
from fairness_pipeline_dev_toolkit.llm_evals.scoring import (
    rate_disparity,
    refusal_score,
)
from fairness_pipeline_dev_toolkit.monitoring.config import DriftConfig
from fairness_pipeline_dev_toolkit.monitoring.drift import FairnessDriftAndAlertEngine


def _group_score_frame(n_per_group: int = 50) -> pd.DataFrame:
    """Two groups, mixed 0/1 scores, plus a transcript column that must be dropped."""
    group_a = pd.DataFrame(
        {
            "group": ["A"] * n_per_group,
            "score": [1, 0] * (n_per_group // 2),
            "response": [f"transcript-A-{i}" for i in range(n_per_group)],
        }
    )
    group_b = pd.DataFrame(
        {
            "group": ["B"] * n_per_group,
            "score": [1] * (n_per_group // 5) + [0] * (n_per_group - n_per_group // 5),
            "response": [f"transcript-B-{i}" for i in range(n_per_group)],
        }
    )
    return pd.concat([group_a, group_b], ignore_index=True)


def test_sample_every_1_keeps_all():
    logs = _group_score_frame(20)
    sampled = sample_production_llm_records(
        logs,
        sample_every=1,
        group_col="group",
        score_col="score",
        random_state=0,
    )
    assert len(sampled) == len(logs)
    assert list(sampled.index) == list(logs.index)


def test_sample_fraction_is_exactly_one_over_n_with_fixed_seed():
    logs = _group_score_frame(100)
    sampled = sample_production_llm_records(
        logs,
        sample_every=10,
        group_col="group",
        score_col="score",
        random_state=42,
    )
    # Exact 1/N without-replacement draw: 200 // 10 = 20.
    assert len(sampled) == len(logs) // 10
    again = sample_production_llm_records(
        logs,
        sample_every=10,
        group_col="group",
        score_col="score",
        random_state=42,
    )
    assert list(sampled.index) == list(again.index)


def test_larger_sample_every_keeps_fewer_rows():
    logs = _group_score_frame(120)
    kept_n2 = sample_production_llm_records(
        logs, sample_every=2, group_col="group", score_col="score", random_state=7
    )
    kept_n5 = sample_production_llm_records(
        logs, sample_every=5, group_col="group", score_col="score", random_state=7
    )
    kept_n10 = sample_production_llm_records(
        logs, sample_every=10, group_col="group", score_col="score", random_state=7
    )
    assert len(kept_n2) == len(logs) // 2
    assert len(kept_n5) == len(logs) // 5
    assert len(kept_n10) == len(logs) // 10
    assert len(kept_n2) > len(kept_n5) > len(kept_n10)


def test_sampled_rows_preserve_original_relative_order():
    logs = _group_score_frame(40)
    logs.index = range(len(logs) - 1, -1, -1)
    sampled = sample_production_llm_records(
        logs,
        sample_every=4,
        group_col="group",
        score_col="score",
        random_state=1,
    )
    orig = list(logs.index)
    kept = list(sampled.index)
    assert kept == [i for i in orig if i in set(kept)]
    assert kept != sorted(kept), "must follow stream order, not sorted index labels"


def test_omitted_random_state_varies_across_calls():
    logs = _group_score_frame(80)
    first = sample_production_llm_records(
        logs, sample_every=4, group_col="group", score_col="score"
    )
    second = sample_production_llm_records(
        logs, sample_every=4, group_col="group", score_col="score"
    )
    assert len(first) == len(logs) // 4
    assert len(second) == len(logs) // 4
    assert list(first.index) != list(second.index)


def test_caller_supplied_seeds_select_different_rows():
    logs = _group_score_frame(80)
    a = sample_production_llm_records(
        logs, sample_every=4, group_col="group", score_col="score", random_state=1
    )
    b = sample_production_llm_records(
        logs, sample_every=4, group_col="group", score_col="score", random_state=2
    )
    assert list(a.index) != list(b.index)


def test_sampler_emits_binarized_scores_without_transcripts():
    rows = []
    for i in range(10):
        rows.append(
            {
                "group": "A" if i < 5 else "B",
                "response": "I cannot help with that." if i % 2 == 0 else "Sure, here you go.",
            }
        )
    sampled = sample_production_llm_records(
        pd.DataFrame(rows),
        sample_every=1,
        group_col="group",
        text_col="response",
        scorer=refusal_score,
        random_state=0,
    )
    assert "response" not in sampled.columns
    assert set(sampled.columns) == {"group", "y_pred"}
    assert set(sampled["y_pred"].unique()) <= {0, 1}


def test_sampled_rows_land_in_tracker_window_unsampled_do_not(tmp_path):
    logs = _group_score_frame(40)
    sampled = sample_production_llm_records(
        logs,
        sample_every=4,
        group_col="group",
        score_col="score",
        random_state=1,
    )
    unsampled = logs.drop(index=sampled.index)
    assert len(sampled) == 80 // 4
    assert not unsampled.empty
    assert "response" not in sampled.columns

    batch = sampled.copy()
    batch["src_index"] = sampled.index.to_numpy()
    tracker = make_production_llm_tracker(window_size=10_000, artifacts_dir=str(tmp_path))
    assert tracker.cfg.min_group_size == DEFAULT_LLM_MIN_GROUP_SIZE
    assert tracker.cfg.metrics == ("demographic_parity",)

    window = ingest_sampled_production_llm(tracker, batch, group_col="group")
    assert len(window) == len(sampled)
    assert set(window["src_index"]) == set(sampled.index)
    assert set(window["src_index"]).isdisjoint(set(unsampled.index))
    assert "response" not in window.columns
    assert "transcript" not in "".join(map(str, tracker.metrics_ts.columns))


def test_tracker_metrics_use_window_not_unsampled_stream(tmp_path):
    logs = _group_score_frame(40)
    sampled = sample_production_llm_records(
        logs,
        sample_every=4,
        group_col="group",
        score_col="score",
        random_state=1,
    )
    tracker = make_production_llm_tracker(
        window_size=len(sampled),
        artifacts_dir=str(tmp_path),
    )
    window = ingest_sampled_production_llm(tracker, sampled, group_col="group")
    assert len(window) == len(sampled)
    assert len(window) < len(logs)

    by_group = {
        str(g): sampled.loc[sampled["group"] == g, "y_pred"].astype(float).tolist()
        for g in sampled["group"].unique()
    }
    expected_dpd = rate_disparity(by_group)
    dpd_rows = tracker.metrics_ts[
        (tracker.metrics_ts["metric"].str.startswith("DP["))
        & (tracker.metrics_ts["group_key"] == "__DPD__")
        & (tracker.metrics_ts["metric"] != "DP[overall]")
    ]
    assert not dpd_rows.empty
    assert dpd_rows["value"].iloc[-1] == expected_dpd

    full_by_group = {
        str(g): logs.loc[logs["group"] == g, "score"].astype(float).tolist()
        for g in logs["group"].unique()
    }
    full_dpd = rate_disparity(full_by_group)
    # Window metric follows the sampled subset, not the unsampled stream.
    assert len(window) != len(logs)
    if full_dpd != expected_dpd:
        assert dpd_rows["value"].iloc[-1] != full_dpd


def test_synthetic_rate_shift_triggers_drift_alert(tmp_path):
    """Reference: equal group rates. Recent: group B jumps. Engine must alert.

    Uses the real ``analyze()`` (not a stub). Groups clear ``min_group_size=5``.
    Each batch equals ``window_size`` so recent snapshots are not diluted by
    the reference rows. Persistence uses ``DP[...]`` names.
    """
    n_per_group = 6
    window_size = n_per_group * 2
    tracker = make_production_llm_tracker(
        window_size=window_size,
        min_group_size=DEFAULT_LLM_MIN_GROUP_SIZE,
        artifacts_dir=str(tmp_path),
    )

    def _batch(b_rate: float) -> pd.DataFrame:
        a_scores = [0] * n_per_group
        n_b_pos = int(round(b_rate * n_per_group))
        b_scores = [1] * n_b_pos + [0] * (n_per_group - n_b_pos)
        return pd.DataFrame(
            {
                "group": ["A"] * n_per_group + ["B"] * n_per_group,
                "score": a_scores + b_scores,
            }
        )

    n_ref, n_shift = 12, 10
    for _ in range(n_ref):
        sampled = sample_production_llm_records(
            _batch(0.0),
            sample_every=1,
            group_col="group",
            score_col="score",
            random_state=0,
        )
        ingest_sampled_production_llm(tracker, sampled, group_col="group")
    for _ in range(n_shift):
        sampled = sample_production_llm_records(
            _batch(1.0),
            sample_every=1,
            group_col="group",
            score_col="score",
            random_state=0,
        )
        ingest_sampled_production_llm(tracker, sampled, group_col="group")

    engine = FairnessDriftAndAlertEngine(DriftConfig(critical_dpd=0.10))
    alerts = engine.analyze(tracker.metrics_ts, window_points=5, ref_points=10)

    assert alerts, "Expected at least one AlertEvent after the synthetic rate shift"
    dpd_alerts = [
        ev
        for ev in alerts
        if ev.metric.startswith("DP") and ev.group_key == "__DPD__" and ev.metric != "DP[overall]"
    ]
    assert (
        dpd_alerts
    ), f"Expected a DP __DPD__ alert; got {[(e.metric, e.group_key) for e in alerts]}"
    assert dpd_alerts[0].metric.startswith("DP[")
