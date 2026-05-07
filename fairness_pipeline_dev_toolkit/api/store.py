from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

_MAX_ENTRIES = 500


class ResultStore:
    """Thread-safe in-memory store for API run results.

    Capped at *maxsize* entries; the oldest entry is evicted when full (LRU).
    """

    def __init__(self, maxsize: int = _MAX_ENTRIES) -> None:
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def put(self, run_id: str, result: dict) -> None:
        with self._lock:
            if run_id in self._store:
                self._store.move_to_end(run_id)
            self._store[run_id] = result
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict oldest

    def get(self, run_id: str) -> Optional[dict]:
        with self._lock:
            if run_id not in self._store:
                return None
            self._store.move_to_end(run_id)  # refresh on read
            return self._store[run_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
