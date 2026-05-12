from __future__ import annotations

from typing import Any, List

from .abtest import FairnessABTestAnalyzer
from .config import DriftConfig, MonitoringSettings, ReportConfig
from .drift import AlertEvent, FairnessDriftAndAlertEngine
from .tracker import ColumnMap, RealTimeFairnessTracker, TrackerConfig

__all__ = [
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


def __getattr__(name: str) -> Any:
    if name == "FairnessReportingDashboard":
        from .dashboard import FairnessReportingDashboard

        return FairnessReportingDashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
