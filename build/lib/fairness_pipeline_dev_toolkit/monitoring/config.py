from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional


@dataclass
class DriftConfig:
    ks_pvalue_cutoff: float = 0.01
    multi_scale: bool = True
    severity_weights: Dict[str, float] = field(default_factory=lambda: {"DP": 1.0, "EO": 1.2})
    min_persist_points: int = 2
    critical_dpd: float = 0.10
    critical_eod: float = 0.10


@dataclass
class ReportConfig:
    k_anonymity: int = 20


@dataclass
class MonitoringSettings:
    artifacts_dir: str = "artifacts/monitoring"
    drift: DriftConfig = field(default_factory=DriftConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def with_overrides(
        self,
        *,
        artifacts_dir: Optional[str] = None,
        drift: Optional[DriftConfig] = None,
        report: Optional[ReportConfig] = None,
    ) -> "MonitoringSettings":
        return MonitoringSettings(
            artifacts_dir=artifacts_dir or self.artifacts_dir,
            drift=drift or replace(self.drift),
            report=report or replace(self.report),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifacts_dir": self.artifacts_dir,
            "drift": asdict(self.drift),
            "report": asdict(self.report),
        }

    def dump(self, path: Optional[str] = None) -> str:
        target = path or os.path.join(self.artifacts_dir, "monitoring_config.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MonitoringSettings":
        drift_cfg = DriftConfig(**payload.get("drift", {}))
        report_cfg = ReportConfig(**payload.get("report", {}))
        return cls(
            artifacts_dir=payload.get("artifacts_dir", "artifacts/monitoring"),
            drift=drift_cfg,
            report=report_cfg,
        )
