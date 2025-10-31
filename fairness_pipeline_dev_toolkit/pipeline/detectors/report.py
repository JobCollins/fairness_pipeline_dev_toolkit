from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class DetectionReport:
    representation: Dict[str, Any]
    disparity: Dict[str, Any]
    proxy: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)
