"""Tests that the fairpipe shim namespace is correct and object identity is preserved."""


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

    assert fairpipe.__version__ == "0.7.2"


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
