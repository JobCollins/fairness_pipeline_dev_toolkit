from __future__ import annotations

import pandas as pd

from fairness_pipeline_dev_toolkit.cli.main import main


def test_validate_with_ci_and_effects(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 0, 1, 0],
            "yhat": [1, 1, 0, 0, 0, 0, 1, 0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    csv_path = tmp_path / "validate.csv"
    df.to_csv(csv_path, index=False)

    calls = {}

    def _stub_report(results, title):
        dp = results["demographic_parity_difference"]
        eq = results["equalized_odds_difference"]
        assert dp.ci is not None and dp.effect_size is not None
        assert eq.ci is not None and eq.effect_size is not None
        calls["count"] = calls.get("count", 0) + 1
        return "stub"

    monkeypatch.setattr(
        "fairness_pipeline_dev_toolkit.cli.main.to_markdown_report",
        _stub_report,
    )

    exit_code = main(
        [
            "validate",
            "--csv",
            str(csv_path),
            "--y-true",
            "y",
            "--y-pred",
            "yhat",
            "--sensitive",
            "group",
            "--min-group-size",
            "1",
            "--with-ci",
            "--with-effects",
            "--bootstrap-B",
            "50",
        ]
    )

    assert exit_code == 0
    assert calls.get("count") == 1
