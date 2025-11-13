import sys
import types
from dataclasses import asdict

from fairness_pipeline_dev_toolkit.integration import mlflow_logger
from fairness_pipeline_dev_toolkit.metrics.base import MetricResult


def test_log_fairness_metrics_uses_string_payloads(monkeypatch):
    log_metrics = []
    log_params = []
    log_dict_calls = []
    log_text_calls = []

    module = types.ModuleType("mlflow")

    def log_metric(name, value):
        log_metrics.append((name, value))

    def log_param(name, value):
        log_params.append((name, value))

    def log_dict(data, artifact_file):
        log_dict_calls.append((data, artifact_file))

    def log_text(text, artifact_file):
        log_text_calls.append((text, artifact_file))

    module.log_metric = log_metric
    module.log_param = log_param
    module.log_dict = log_dict
    module.log_text = log_text

    monkeypatch.setitem(sys.modules, "mlflow", module)
    monkeypatch.setattr(mlflow_logger, "_is_mlflow_available", lambda: True)

    metric = MetricResult(
        metric="demographic_parity_difference",
        value=0.12,
        ci=(0.01, 0.23),
        effect_size=0.45,
        n_per_group={"A": 50, "B": 60},
    )

    ok = mlflow_logger.log_fairness_metrics(
        {"dpd": metric}, artifact_name="fairness_report.md", artifact_content="# report"
    )

    assert ok
    assert log_metrics == [("fairness_.dpd.value", 0.12)]
    assert any(name.endswith(".ci") for name, _ in log_params)
    assert log_dict_calls == [({"dpd": asdict(metric)}, "fairness_results.json")]
    assert log_text_calls == [("# report", "fairness_report.md")]
