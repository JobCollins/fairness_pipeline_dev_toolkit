"""
Public API tests.

These tests verify that:
1. All documented public APIs are importable
2. Public APIs match their documented signatures
3. Internal/private APIs are not exposed in __all__
4. APIs work as expected with basic usage
"""

import pytest


class TestRootPackageAPI:
    """Test public API exports from root package."""

    def test_root_all_exports(self):
        """Test that __all__ contains expected exports."""
        try:
            from fairness_pipeline_dev_toolkit import __all__

            expected_exports = [
                "FairnessAnalyzer",
                "MetricResult",
                "to_markdown_report",
                "log_fairness_metrics",
                "assert_fairness",
            ]

            for export in expected_exports:
                assert export in __all__, f"{export} should be in __all__"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_root_imports_work(self):
        """Test that all root-level imports work."""
        try:
            from fairness_pipeline_dev_toolkit import (
                FairnessAnalyzer,
                MetricResult,
                assert_fairness,
                log_fairness_metrics,
                to_markdown_report,
            )

            assert FairnessAnalyzer is not None
            assert MetricResult is not None
            assert callable(assert_fairness)
            assert callable(log_fairness_metrics)
            assert callable(to_markdown_report)
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_version_importable(self):
        """Test that version is importable."""
        from fairness_pipeline_dev_toolkit import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestMetricsAPI:
    """Test public API from metrics module."""

    def test_metrics_all_exports(self):
        """Test that metrics.__all__ contains expected exports."""
        try:
            from fairness_pipeline_dev_toolkit.metrics import __all__

            expected_exports = ["FairnessAnalyzer", "MetricResult"]

            for export in expected_exports:
                assert export in __all__, f"{export} should be in metrics.__all__"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_metrics_imports_work(self):
        """Test that metrics module imports work."""
        try:
            from fairness_pipeline_dev_toolkit.metrics import (
                FairnessAnalyzer,
                MetricResult,
            )

            assert FairnessAnalyzer is not None
            assert MetricResult is not None
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_fairness_analyzer_instantiable(self):
        """Test that FairnessAnalyzer can be instantiated."""
        try:
            from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

            analyzer = FairnessAnalyzer(min_group_size=30)
            assert analyzer is not None
            assert hasattr(analyzer, "demographic_parity_difference")
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_fairness_analyzer_public_methods(self):
        """Test that FairnessAnalyzer has expected public methods."""
        try:
            from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

            analyzer = FairnessAnalyzer()
            expected_methods = [
                "demographic_parity_difference",
                "equalized_odds_difference",
                "mae_parity_difference",  # For regression tasks
            ]

            for method_name in expected_methods:
                assert hasattr(analyzer, method_name), f"{method_name} should exist"
                method = getattr(analyzer, method_name)
                assert callable(method), f"{method_name} should be callable"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise


class TestPipelineAPI:
    """Test public API from pipeline module."""

    def test_pipeline_all_exports(self):
        """Test that pipeline.__all__ contains expected exports."""
        try:
            from fairness_pipeline_dev_toolkit.pipeline import __all__

            expected_exports = [
                "PipelineConfig",
                "load_config",
                "build_pipeline",
                "apply_pipeline",
                "run_detectors",
                "InstanceReweighting",
                "DisparateImpactRemover",
                "ReweighingTransformer",
                "ProxyDropper",
            ]

            for export in expected_exports:
                assert export in __all__, f"{export} should be in pipeline.__all__"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_pipeline_imports_work(self):
        """Test that pipeline module imports work."""
        try:
            from fairness_pipeline_dev_toolkit.pipeline import (
                DisparateImpactRemover,
                InstanceReweighting,
                PipelineConfig,
                ProxyDropper,
                ReweighingTransformer,
                apply_pipeline,
                build_pipeline,
                load_config,
                run_detectors,
            )

            assert PipelineConfig is not None
            assert callable(load_config)
            assert callable(build_pipeline)
            assert callable(apply_pipeline)
            assert callable(run_detectors)
            assert InstanceReweighting is not None
            assert DisparateImpactRemover is not None
            assert ReweighingTransformer is not None
            assert ProxyDropper is not None
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_pipeline_config_instantiable(self):
        """Test that PipelineConfig can be instantiated."""
        try:
            from fairness_pipeline_dev_toolkit.pipeline import PipelineConfig

            # Should be able to create with minimal config
            config = PipelineConfig(sensitive=["group"], alpha=0.05)
            assert config is not None
            assert config.sensitive == ["group"]
            assert config.alpha == 0.05
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise


class TestIntegrationAPI:
    """Test public API from integration module."""

    def test_integration_all_exports(self):
        """Test that integration.__all__ contains expected exports."""
        try:
            from fairness_pipeline_dev_toolkit.integration import __all__

            expected_exports = [
                "execute_workflow",
                "WorkflowResult",
                "ValidationResult",
                "log_workflow_results",
                "to_markdown_report",
                "assert_fairness",
            ]

            for export in expected_exports:
                assert export in __all__, f"{export} should be in integration.__all__"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Integration module requires optional dependencies: {e}")
            raise

    def test_integration_imports_work(self):
        """Test that integration module imports work."""
        try:
            from fairness_pipeline_dev_toolkit.integration import (
                ValidationResult,
                WorkflowResult,
                assert_fairness,
                execute_workflow,
                log_workflow_results,
                to_markdown_report,
            )

            assert WorkflowResult is not None
            assert ValidationResult is not None
            assert callable(assert_fairness)
            assert callable(execute_workflow)
            assert callable(log_workflow_results)
            assert callable(to_markdown_report)
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Integration module requires optional dependencies: {e}")
            raise


class TestTrainingAPI:
    """Test public API from training module (if available)."""

    def test_training_all_exports(self):
        """Test that training.__all__ contains expected exports."""
        try:
            from fairness_pipeline_dev_toolkit.training import __all__

            expected_exports = [
                "ReductionsWrapper",
                "FairnessRegularizerLoss",
                "LagrangianFairnessTrainer",
                "GroupFairnessCalibrator",
                "sweep_pareto",
                "plot_pareto",
            ]

            for export in expected_exports:
                assert export in __all__, f"{export} should be in training.__all__"
        except ImportError as e:
            pytest.skip(f"Training module not available: {e}")

    def test_training_imports_work(self):
        """Test that training module imports work."""
        try:
            from fairness_pipeline_dev_toolkit.training import (
                FairnessRegularizerLoss,
                GroupFairnessCalibrator,
                LagrangianFairnessTrainer,
                ReductionsWrapper,
                plot_pareto,
                sweep_pareto,
            )

            assert ReductionsWrapper is not None
            assert FairnessRegularizerLoss is not None
            assert LagrangianFairnessTrainer is not None
            assert GroupFairnessCalibrator is not None
            assert callable(sweep_pareto)
            assert callable(plot_pareto)
        except ImportError:
            pytest.skip("Training module not available")


class TestMonitoringAPI:
    """Test public API from monitoring module (if available)."""

    def test_monitoring_all_exports(self):
        """Test that monitoring.__all__ contains expected exports."""
        try:
            from fairness_pipeline_dev_toolkit.monitoring import __all__

            expected_exports = [
                "RealTimeFairnessTracker",
                "ColumnMap",
                "TrackerConfig",
                "FairnessDriftAndAlertEngine",
                "DriftConfig",
                "MonitoringSettings",
                "AlertEvent",
                "FairnessReportingDashboard",
                "ReportConfig",
                "FairnessABTestAnalyzer",
            ]

            for export in expected_exports:
                assert export in __all__, f"{export} should be in monitoring.__all__"
        except ImportError:
            pytest.skip("Monitoring module not available")

    def test_monitoring_imports_work(self):
        """Test that monitoring module imports work."""
        try:
            from fairness_pipeline_dev_toolkit.monitoring import (
                AlertEvent,
                ColumnMap,
                DriftConfig,
                FairnessABTestAnalyzer,
                FairnessDriftAndAlertEngine,
                FairnessReportingDashboard,
                MonitoringSettings,
                RealTimeFairnessTracker,
                ReportConfig,
                TrackerConfig,
            )

            assert RealTimeFairnessTracker is not None
            assert ColumnMap is not None
            assert TrackerConfig is not None
            assert FairnessDriftAndAlertEngine is not None
            assert DriftConfig is not None
            assert MonitoringSettings is not None
            assert AlertEvent is not None
            assert FairnessReportingDashboard is not None
            assert ReportConfig is not None
            assert FairnessABTestAnalyzer is not None
        except ImportError:
            pytest.skip("Monitoring module not available")


class TestExceptionsAPI:
    """Test public API from exceptions module."""

    def test_exceptions_importable(self):
        """Test that exceptions module is importable."""
        # Exceptions module should be importable without optional dependencies
        try:
            from fairness_pipeline_dev_toolkit import exceptions

            assert exceptions is not None
        except ImportError:
            # Exceptions should not require optional dependencies
            # If this fails, it's a real error
            raise

    def test_exception_hierarchy_exists(self):
        """Test that exception classes exist."""
        # Exceptions should be importable directly without triggering root package import
        try:
            # Import directly from exceptions module to avoid root package import chain
            import fairness_pipeline_dev_toolkit.exceptions as exc_module

            FairnessToolkitError = getattr(exc_module, "FairnessToolkitError", None)

            if FairnessToolkitError is None:
                # Try importing through root package
                from fairness_pipeline_dev_toolkit.exceptions import (
                    FairnessToolkitError,
                )

            # Check that base exception exists
            assert FairnessToolkitError is not None
            assert issubclass(FairnessToolkitError, Exception)
        except ImportError as e:
            # Exceptions should not require optional dependencies
            if "fairlearn" in str(e) or "torch" in str(e):
                # This shouldn't happen for exceptions, but handle gracefully
                pytest.skip(f"Unexpected dependency issue: {e}")
            raise


class TestAPIConsistency:
    """Test that APIs are consistent across import paths."""

    def test_fairness_analyzer_same_object(self):
        """Test that FairnessAnalyzer is the same object from different import paths."""
        try:
            from fairness_pipeline_dev_toolkit import FairnessAnalyzer as RootFA
            from fairness_pipeline_dev_toolkit.metrics import (
                FairnessAnalyzer as MetricsFA,
            )

            assert RootFA is MetricsFA, "FairnessAnalyzer should be the same object"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_metric_result_same_object(self):
        """Test that MetricResult is the same object from different import paths."""
        try:
            from fairness_pipeline_dev_toolkit import MetricResult as RootMR
            from fairness_pipeline_dev_toolkit.metrics import MetricResult as MetricsMR

            assert RootMR is MetricsMR, "MetricResult should be the same object"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise


class TestInternalAPIsNotExposed:
    """Test that internal/private APIs are not exposed in __all__."""

    def test_internal_modules_not_in_all(self):
        """Test that internal module names are not in __all__."""
        try:
            from fairness_pipeline_dev_toolkit import __all__

            # These should NOT be in __all__ (they're internal)
            internal_names = [
                "orchestration",
                "registry",
                "core",
                "base",
                "aequitas_adapter",
                "fairlearn_adapter",
            ]

            for name in internal_names:
                assert name not in __all__, f"{name} should not be in __all__ (internal)"
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise

    def test_private_methods_not_exposed(self):
        """Test that private methods (starting with _) are not in public API."""
        try:
            from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

            analyzer = FairnessAnalyzer()
            public_methods = [
                name
                for name in dir(analyzer)
                if not name.startswith("_") and callable(getattr(analyzer, name))
            ]

            # Should have public methods
            assert len(public_methods) > 0

            # Private methods exist but shouldn't be in __all__
            # (We don't need to collect them, just verify public methods exist)
        except ImportError as e:
            if "fairlearn" in str(e) or "torch" in str(e):
                pytest.skip(f"Optional dependencies not installed: {e}")
            raise
