"""Tests that the fairpipe shim namespace is correct and object identity is preserved."""


def test_fairpipe_measurement_shim():
    """`fairpipe.measurement` mirrors the toolkit facade; no optional monitoring deps."""
    from fairpipe.measurement import (
        FairnessAnalyzer,
        assert_fairness,
        to_markdown_report,
    )
    from fairpipe.metrics import FairnessAnalyzer as FairnessAnalyzer_metrics

    assert FairnessAnalyzer is FairnessAnalyzer_metrics
    assert callable(assert_fairness)
    assert callable(to_markdown_report)


def test_fairpipe_integration_exports_log_fairness_metrics():
    from fairpipe.integration import execute_workflow, log_fairness_metrics

    assert callable(log_fairness_metrics)
    assert callable(execute_workflow)


def test_fairpipe_stats_shim_identity():
    from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci as src_ci
    from fairpipe.stats.bootstrap import bootstrap_ci as shim_ci

    assert shim_ci is src_ci


def test_fairpipe_api_shim_identity():
    from fairness_pipeline_dev_toolkit.api import create_app as src_create
    from fairpipe.api import ResultStore
    from fairpipe.api import create_app as shim_create

    assert shim_create is src_create
    assert ResultStore is not None


def test_top_level_imports():
    import fairpipe  # noqa: F401
    from fairpipe import FairnessAnalyzer, MetricResult  # noqa: F401
    from fairpipe.exceptions import (  # noqa: F401
        ConfigValidationError,
        FairnessToolkitError,
        MetricComputationError,
        PipelineExecutionError,
    )
    from fairpipe.integration import (  # noqa: F401
        ValidationResult,
        WorkflowResult,
        execute_workflow,
    )
    from fairpipe.monitoring import RealTimeFairnessTracker  # noqa: F401
    from fairpipe.monitoring import (  # noqa: F401
        FairnessDriftAndAlertEngine,
        FairnessReportingDashboard,
    )
    from fairpipe.pipeline import (  # noqa: F401
        DisparateImpactRemover,
        InstanceReweighting,
        PipelineResult,
        ProxyDropper,
        ReweighingTransformer,
        apply_pipeline,
        build_pipeline,
        load_config,
        run_detectors,
    )
    from fairpipe.training import (  # noqa: F401
        FairnessRegularizerLoss,
        GroupFairnessCalibrator,
        LagrangianFairnessTrainer,
        ReductionsWrapper,
    )


def test_same_objects():
    from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer as B
    from fairpipe import FairnessAnalyzer as A

    assert A is B


def test_version_accessible():
    import fairpipe
    from fairness_pipeline_dev_toolkit import __version__ as src_version

    assert isinstance(fairpipe.__version__, str)
    assert fairpipe.__version__ == src_version


def test_fairpipe_llm_evals_shim_identity():
    from fairness_pipeline_dev_toolkit.llm_evals import LLMEvalAdapter as src
    from fairness_pipeline_dev_toolkit.llm_evals import (
        sample_production_llm_records as src_sample,
    )
    from fairpipe.llm_evals import LLMEvalAdapter as shim
    from fairpipe.llm_evals import sample_production_llm_records as shim_sample

    assert shim is src
    assert shim_sample is src_sample


def test_legacy_imports_still_work():
    from fairness_pipeline_dev_toolkit import (  # noqa: F401
        FairnessAnalyzer,
        MetricResult,
    )
    from fairness_pipeline_dev_toolkit.exceptions import (  # noqa: F401
        ConfigValidationError,
        FairnessToolkitError,
        MetricComputationError,
        PipelineExecutionError,
    )
    from fairness_pipeline_dev_toolkit.integration import (  # noqa: F401
        ValidationResult,
        WorkflowResult,
        execute_workflow,
    )
    from fairness_pipeline_dev_toolkit.monitoring import (  # noqa: F401
        FairnessDriftAndAlertEngine,
        FairnessReportingDashboard,
        RealTimeFairnessTracker,
    )
    from fairness_pipeline_dev_toolkit.pipeline import (  # noqa: F401
        DisparateImpactRemover,
        InstanceReweighting,
        PipelineResult,
        ProxyDropper,
        ReweighingTransformer,
        apply_pipeline,
        build_pipeline,
        load_config,
        run_detectors,
    )
    from fairness_pipeline_dev_toolkit.training import (  # noqa: F401
        FairnessRegularizerLoss,
        GroupFairnessCalibrator,
        LagrangianFairnessTrainer,
        ReductionsWrapper,
    )
