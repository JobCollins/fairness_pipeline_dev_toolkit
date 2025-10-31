from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, ItemsView, Iterator, KeysView, Optional, ValuesView


@dataclass
class DetectionReport:
    representation: Dict[str, Any]
    disparity: Dict[str, Any]
    proxy: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BiasReport(Mapping[str, Any]):
    """
    Minimal container for detector output that supports attribute access
    (e.g., rep.meta) and also behaves like a read-only dict so existing
    code that does `report.items()` keeps working.
    """

    meta: Dict[str, Any] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Keep 'meta' explicit at the top level for artifact clarity
        return {"meta": self.meta, "body": self.body}

    def to_json(self, *, indent: int = 2, ensure_ascii: bool = False) -> str:
        """Stringified JSON convenience."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    # ---- Mapping (dict-like) interface ----
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    # Optional explicit views (not strictly required, but nice to have)
    def items(self) -> ItemsView[str, Any]:
        return self.to_dict().items()

    def keys(self) -> KeysView[str]:
        return self.to_dict().keys()

    def values(self) -> ValuesView[Any]:
        return self.to_dict().values()

    # Friendly alias if you prefer explicit naming elsewhere
    def asdict(self) -> Dict[str, Any]:
        return self.to_dict()

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
