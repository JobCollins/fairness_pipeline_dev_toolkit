"""
Unit tests for pipeline detector stubs.

Tests the scaffold stub implementations:
- DisparityDetector
- ProxyDetector
- RepresentationDetector
"""

from __future__ import annotations

import pandas as pd

from fairness_pipeline_dev_toolkit.pipeline.detectors.disparity import (
    DisparityDetector,
    DisparityResult,
)
from fairness_pipeline_dev_toolkit.pipeline.detectors.proxy import (
    ProxyDetector,
    ProxyResult,
)
from fairness_pipeline_dev_toolkit.pipeline.detectors.representation import (
    RepresentationDetector,
    RepresentationResult,
)

# ============================================================================
# DisparityDetector Tests
# ============================================================================


class TestDisparityDetector:
    """Tests for DisparityDetector stub implementation."""

    def test_run_returns_disparity_result_with_empty_metrics(self):
        """Test DisparityDetector.run() returns DisparityResult with empty metrics dict."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "target": [0, 1, 0, 1],
                "feature": [1.0, 2.0, 3.0, 4.0],
            }
        )

        detector = DisparityDetector()
        result = detector.run(df, target="target", sensitive=["group"])

        assert isinstance(result, DisparityResult)
        assert result.metrics == {}
        assert result.notes == "Scafold stub"

    def test_run_with_none_target(self):
        """Test DisparityDetector.run() with None target."""
        df = pd.DataFrame({"group": ["A", "B"], "feature": [1.0, 2.0]})

        detector = DisparityDetector()
        result = detector.run(df, target=None, sensitive=["group"])

        assert isinstance(result, DisparityResult)
        assert result.metrics == {}
        assert result.notes == "Scafold stub"

    def test_run_with_multiple_sensitive_attributes(self):
        """Test DisparityDetector.run() with multiple sensitive attributes."""
        df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "gender": ["M", "F"],
                "target": [0, 1],
            }
        )

        detector = DisparityDetector()
        result = detector.run(df, target="target", sensitive=["group", "gender"])

        assert isinstance(result, DisparityResult)
        assert result.metrics == {}


# ============================================================================
# ProxyDetector Tests
# ============================================================================


class TestProxyDetector:
    """Tests for ProxyDetector stub implementation."""

    def test_run_returns_proxy_result_with_empty_associations(self):
        """Test ProxyDetector.run() returns ProxyResult with empty associations dict."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "feature2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        detector = ProxyDetector()
        result = detector.run(df, features=["feature1", "feature2"], sensitive=["group"])

        assert isinstance(result, ProxyResult)
        assert result.associations == {}
        assert result.notes == "Scafold stub"

    def test_run_with_single_feature(self):
        """Test ProxyDetector.run() with single feature."""
        df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "feature": [1.0, 2.0],
            }
        )

        detector = ProxyDetector()
        result = detector.run(df, features=["feature"], sensitive=["group"])

        assert isinstance(result, ProxyResult)
        assert result.associations == {}

    def test_run_with_multiple_sensitive_attributes(self):
        """Test ProxyDetector.run() with multiple sensitive attributes."""
        df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "gender": ["M", "F"],
                "feature": [1.0, 2.0],
            }
        )

        detector = ProxyDetector()
        result = detector.run(df, features=["feature"], sensitive=["group", "gender"])

        assert isinstance(result, ProxyResult)
        assert result.associations == {}


# ============================================================================
# RepresentationDetector Tests
# ============================================================================


class TestRepresentationDetector:
    """Tests for RepresentationDetector stub implementation."""

    def test_run_returns_representation_result_with_empty_by_group(self):
        """Test RepresentationDetector.run() returns RepresentationResult with empty by_group dict."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C"],
                "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        detector = RepresentationDetector()
        result = detector.run(df, sensitive=["group"])

        assert isinstance(result, RepresentationResult)
        assert result.by_group == {}
        assert result.notes == "Scaffold stub"
        assert result.benchmarks is None

    def test_run_with_benchmarks(self):
        """Test RepresentationDetector.run() with benchmarks parameter."""
        df = pd.DataFrame({"group": ["A", "B"], "feature": [1.0, 2.0]})
        benchmarks = {"group": {"A": 0.5, "B": 0.5}}

        detector = RepresentationDetector()
        result = detector.run(df, sensitive=["group"], benchmarks=benchmarks)

        assert isinstance(result, RepresentationResult)
        assert result.by_group == {}
        assert result.benchmarks == benchmarks

    def test_run_with_none_benchmarks(self):
        """Test RepresentationDetector.run() with None benchmarks."""
        df = pd.DataFrame({"group": ["A", "B"], "feature": [1.0, 2.0]})

        detector = RepresentationDetector()
        result = detector.run(df, sensitive=["group"], benchmarks=None)

        assert isinstance(result, RepresentationResult)
        assert result.by_group == {}
        assert result.benchmarks is None

    def test_run_with_multiple_sensitive_attributes(self):
        """Test RepresentationDetector.run() with multiple sensitive attributes."""
        df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "gender": ["M", "F"],
                "feature": [1.0, 2.0],
            }
        )

        detector = RepresentationDetector()
        result = detector.run(df, sensitive=["group", "gender"])

        assert isinstance(result, RepresentationResult)
        assert result.by_group == {}
