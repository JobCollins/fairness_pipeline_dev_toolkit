"""Unit and property tests for KamiranCaldersReweighing."""

from __future__ import annotations

import io
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairness_pipeline_dev_toolkit.exceptions import KamiranCaldersLabelError
from fairness_pipeline_dev_toolkit.pipeline.config import load_config
from fairness_pipeline_dev_toolkit.pipeline.orchestration import (
    apply_pipeline,
    build_pipeline,
)
from fairness_pipeline_dev_toolkit.pipeline.transformers import (
    InstanceReweighting,
    KamiranCaldersReweighing,
    ReweighingTransformer,
)


def _oracle_frame() -> pd.DataFrame:
    """100-row Kamiran–Calders worked example.

    Group A: 40 positive, 20 negative; group B: 10 positive, 30 negative.
    """
    rows = (
        [{"group": "A", "y": 1, "f": i} for i in range(40)]
        + [{"group": "A", "y": 0, "f": i} for i in range(40, 60)]
        + [{"group": "B", "y": 1, "f": i} for i in range(60, 70)]
        + [{"group": "B", "y": 0, "f": i} for i in range(70, 100)]
    )
    return pd.DataFrame(rows)


def _weighted_positive_rate(groups, labels, weights) -> dict:
    out = {}
    g_list = list(groups)
    y = np.asarray(labels)
    w = np.asarray(weights, dtype=float)
    seen = []
    for gi in g_list:
        if gi not in seen:
            seen.append(gi)
    for name in seen:
        m = np.array([gi == name for gi in g_list], dtype=bool)
        denom = w[m].sum()
        out[name] = float(w[m & (y == 1)].sum() / denom) if denom > 0 else float("nan")
    return out


class TestKamiranCaldersOracle:
    def test_oracle_weights_and_independence(self):
        df = _oracle_frame()
        kc = KamiranCaldersReweighing(sensitive=["group"]).fit(df.drop(columns=["y"]), df["y"])
        w = kc.sample_weight_

        # Exact cell weights: 0.75 / 1.5 / 2.0 / 2/3
        assert kc.cell_weights_[("A", 1)] == pytest.approx(0.75)
        assert kc.cell_weights_[("A", 0)] == pytest.approx(1.5)
        assert kc.cell_weights_[("B", 1)] == pytest.approx(2.0)
        assert kc.cell_weights_[("B", 0)] == pytest.approx(2.0 / 3.0)

        expected = np.where(
            (df["group"] == "A") & (df["y"] == 1),
            0.75,
            np.where(
                (df["group"] == "A") & (df["y"] == 0),
                1.5,
                np.where((df["group"] == "B") & (df["y"] == 1), 2.0, 2.0 / 3.0),
            ),
        )
        np.testing.assert_allclose(w, expected)
        assert w.sum() == pytest.approx(100.0)

        rates = _weighted_positive_rate(df["group"], df["y"], w)
        assert rates["A"] == pytest.approx(0.5)
        assert rates["B"] == pytest.approx(0.5)


class TestKamiranCaldersVsFrequency:
    def test_kc_removes_base_rate_gap_frequency_does_not(self):
        df = _oracle_frame()
        # Unweighted gap: 40/60 - 10/40 = 2/3 - 1/4 = 5/12 ≈ 0.4167
        raw_a = (df.loc[df["group"] == "A", "y"] == 1).mean()
        raw_b = (df.loc[df["group"] == "B", "y"] == 1).mean()
        raw_gap = abs(raw_a - raw_b)
        assert raw_gap == pytest.approx(5.0 / 12.0)

        ir = InstanceReweighting(sensitive=["group"]).fit(df)
        # Frequency weights are constant within group → base rates unchanged.
        rates_ir = _weighted_positive_rate(df["group"], df["y"], ir.sample_weight_)
        assert abs(rates_ir["A"] - rates_ir["B"]) == pytest.approx(raw_gap, abs=1e-9)

        kc = KamiranCaldersReweighing(sensitive=["group"]).fit(df.drop(columns=["y"]), df["y"])
        rates_kc = _weighted_positive_rate(df["group"], df["y"], kc.sample_weight_)
        assert rates_kc["A"] == pytest.approx(rates_kc["B"], abs=1e-12)


class TestKamiranCaldersProperties:
    def test_random_joint_groups_independence(self):
        rng = np.random.default_rng(0)
        n = 400
        sex = rng.choice(["F", "M"], size=n)
        race = rng.choice(["X", "Y", "Z"], size=n, p=[0.5, 0.3, 0.2])
        # Label depends on joint group so unweighted rates differ.
        p = np.where((sex == "F") & (race == "X"), 0.8, 0.3)
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({"sex": sex, "race": race, "f": rng.normal(size=n), "y": y})

        kc = KamiranCaldersReweighing(sensitive=["sex", "race"]).fit(
            df.drop(columns=["y"]), df["y"]
        )
        w = kc.sample_weight_
        assert w.sum() == pytest.approx(float(n))

        joint = list(zip(df["sex"], df["race"]))
        rates = _weighted_positive_rate(joint, df["y"], w)
        # All joint cells with mass share the same weighted positive rate (= overall).
        overall = float(np.average(df["y"], weights=w))
        for rate in rates.values():
            assert rate == pytest.approx(overall, abs=1e-9)

    def test_joint_not_per_attribute_product(self):
        """Independence holds on intersections, not only marginals."""
        # Construct so product-of-marginals would differ from joint KC.
        rows = []
        # sex×race cells with unequal base rates
        for sex, race, n_pos, n_neg in [
            ("F", "X", 30, 10),
            ("F", "Y", 5, 25),
            ("M", "X", 5, 25),
            ("M", "Y", 20, 10),
        ]:
            rows.extend(
                [{"sex": sex, "race": race, "y": 1, "f": 0}] * n_pos
                + [{"sex": sex, "race": race, "y": 0, "f": 0}] * n_neg
            )
        df = pd.DataFrame(rows)
        kc = KamiranCaldersReweighing(sensitive=["sex", "race"]).fit(
            df.drop(columns=["y"]), df["y"]
        )
        joint = list(zip(df["sex"], df["race"]))
        rates = _weighted_positive_rate(joint, df["y"], kc.sample_weight_)
        vals = list(rates.values())
        assert all(v == pytest.approx(vals[0], abs=1e-9) for v in vals)


class TestKamiranCaldersLabelPaths:
    def test_y_arg_label_column_and_config_default_match(self, tmp_path: Path):
        df = _oracle_frame()
        X = df.drop(columns=["y"])
        y = df["y"]

        w_y = KamiranCaldersReweighing(sensitive=["group"]).fit(X, y).sample_weight_
        X_with_y = df.copy()
        w_col = (
            KamiranCaldersReweighing(sensitive=["group"], label="y").fit(X_with_y).sample_weight_
        )

        cfg = load_config(
            text=textwrap.dedent(
                """
                sensitive: ["group"]
                pipeline:
                  - name: kc
                    transformer: "KamiranCaldersReweighing"
                    params: {}
                training:
                  method: "reductions"
                  target_column: "y"
                """
            )
        )
        pipe = build_pipeline(cfg)
        # Config defaults label to training.target_column; fit without y uses X["y"].
        pr = apply_pipeline(pipe, X_with_y, fit=True)
        w_cfg = pr.sample_weight

        np.testing.assert_allclose(w_y, w_col)
        np.testing.assert_allclose(w_y, w_cfg)

    def test_missing_labels_raises_named_error(self):
        df = _oracle_frame().drop(columns=["y"])
        with pytest.raises(KamiranCaldersLabelError, match="requires binary labels"):
            KamiranCaldersReweighing(sensitive=["group"]).fit(df)


class TestKamiranCaldersLabelRejection:
    def test_rejects_multiclass(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "f": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 2, 1])
        with pytest.raises(KamiranCaldersLabelError, match="binary"):
            KamiranCaldersReweighing(sensitive=["group"]).fit(df, y)

    def test_rejects_non_binary_encoding(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "f": [1, 2, 3, 4]})
        y = pd.Series([-1, 1, -1, 1])
        with pytest.raises(KamiranCaldersLabelError, match=r"\{0, 1\}"):
            KamiranCaldersReweighing(sensitive=["group"]).fit(df, y)

    def test_rejects_nan_labels(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "f": [1, 2, 3, 4]})
        y = pd.Series([0, 1, np.nan, 1])
        with pytest.raises(KamiranCaldersLabelError, match="NaN"):
            KamiranCaldersReweighing(sensitive=["group"]).fit(df, y)


class TestKamiranCaldersDegenerateAndClip:
    def test_degenerate_group_warns_and_records(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "B"],
                "f": [1, 2, 3, 4, 5],
                "y": [1, 1, 0, 1, 0],  # A is all-positive
            }
        )
        with pytest.warns(UserWarning, match="single label"):
            kc = KamiranCaldersReweighing(sensitive=["group"]).fit(df.drop(columns=["y"]), df["y"])
        assert "A" in kc.degenerate_groups_
        assert kc.sample_weight_ is not None
        assert len(kc.sample_weight_) == len(df)

    def test_max_weight_clips_and_renormalizes(self):
        df = _oracle_frame()
        w_free = (
            KamiranCaldersReweighing(sensitive=["group"])
            .fit(df.drop(columns=["y"]), df["y"])
            .sample_weight_
        )
        kc = KamiranCaldersReweighing(sensitive=["group"], max_weight=1.2).fit(
            df.drop(columns=["y"]), df["y"]
        )
        w = kc.sample_weight_
        # Clip then mean-normalize (post-renorm max may exceed max_weight).
        assert w.mean() == pytest.approx(1.0)
        assert np.all(np.isfinite(w))
        assert np.all(w > 0)
        assert not np.allclose(w, w_free)


class TestKamiranCaldersTrainOnly:
    def test_transform_held_out_does_not_change_weights_or_features(self):
        df = _oracle_frame()
        train = df.iloc[:80].reset_index(drop=True)
        test = df.iloc[80:].reset_index(drop=True)
        kc = KamiranCaldersReweighing(sensitive=["group"]).fit(
            train.drop(columns=["y"]), train["y"]
        )
        w_before = kc.sample_weight_.copy()
        out = kc.transform(test.drop(columns=["y"]))
        pd.testing.assert_frame_equal(out, test.drop(columns=["y"]))
        np.testing.assert_array_equal(kc.sample_weight_, w_before)
        assert len(kc.sample_weight_) == len(train)


class TestKamiranCaldersSensitiveNan:
    def test_nan_sensitive_rows_get_weight_one(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", np.nan, "B"],
                "f": [1, 2, 3, 4, 5],
                "y": [1, 0, 1, 1, 0],
            }
        )
        kc = KamiranCaldersReweighing(sensitive=["group"]).fit(df.drop(columns=["y"]), df["y"])
        assert kc.n_excluded_sensitive_nan_ == 1
        assert kc.sample_weight_[3] == pytest.approx(1.0)


class TestReweighingTransformerDeprecation:
    def test_emits_future_warning(self):
        with pytest.warns(FutureWarning, match="InstanceReweighting"):
            ReweighingTransformer(sensitive=["group"])

    def test_weights_unchanged_regression_guard(self):
        """Pinned weights from the pre-deprecation frequency-balancer formula."""
        df = pd.DataFrame(
            {
                "feature1": np.arange(10),
                "group": ["A"] * 7 + ["B"] * 3,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            transformer = ReweighingTransformer(sensitive=["group"], clip=10.0)
            transformer.fit(df)
        # Uniform targets over 2 groups → target 0.5 each.
        # obs A=0.7 → w=0.5/0.7; obs B=0.3 → w=0.5/0.3; then mean-normalize.
        raw = np.array([0.5 / 0.7] * 7 + [0.5 / 0.3] * 3)
        expected = raw / raw.mean()
        np.testing.assert_allclose(transformer.sample_weight_, expected)


class TestKamiranCaldersCLIAndREST:
    def test_cli_pipeline_kc_without_label_exits_2(self, tmp_path: Path):
        from fairness_pipeline_dev_toolkit.cli.main import cmd_pipeline_run

        csv_path = tmp_path / "data.csv"
        cfg_path = tmp_path / "cfg.yml"
        _oracle_frame().drop(columns=["y"]).to_csv(csv_path, index=False)
        cfg_path.write_text(
            textwrap.dedent(
                """
                sensitive: ["group"]
                pipeline:
                  - name: kc
                    transformer: "KamiranCaldersReweighing"
                    params: {}
                """
            ),
            encoding="utf-8",
        )

        class Args:
            config = str(cfg_path)
            csv = str(csv_path)
            profile = None
            no_detectors = True
            detector_json = None
            out_csv = None
            report_md = None

        assert cmd_pipeline_run(Args()) == 2

    def test_rest_pipeline_kc_without_label_returns_422(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from fairness_pipeline_dev_toolkit.api import create_app

        app = create_app()
        client = TestClient(app)
        df = _oracle_frame().drop(columns=["y"])
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        config = textwrap.dedent(
            """
            sensitive: ["group"]
            pipeline:
              - name: kc
                transformer: "KamiranCaldersReweighing"
                params: {}
            """
        )
        r = client.post(
            "/pipeline",
            files={"file": ("data.csv", buf.getvalue(), "text/csv")},
            data={"config": config},
        )
        assert r.status_code == 422
        assert "label" in r.json()["detail"].lower() or "Kamiran" in r.json()["detail"]
