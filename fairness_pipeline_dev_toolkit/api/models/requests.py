from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, model_validator


class ValidateRequest(BaseModel):
    y_pred: List[Union[int, float]]
    sensitive: List[Union[str, int]]
    y_true: Optional[List[Union[int, float]]] = None
    y_score: Optional[List[float]] = None
    with_ci: bool = False
    ci_level: float = 0.95
    with_effects: bool = False
    min_group_size: int = 30
    backend: Optional[str] = None
    threshold: Optional[float] = None

    @model_validator(mode="after")
    def check_lengths(self) -> "ValidateRequest":
        n = len(self.y_pred)
        m = len(self.sensitive)
        if m != n:
            raise ValueError(f"y_pred length ({n}) must equal sensitive length ({m})")
        if self.y_true is not None and len(self.y_true) != n:
            raise ValueError(f"y_true length ({len(self.y_true)}) must equal y_pred length ({n})")
        if self.y_score is not None and len(self.y_score) != n:
            raise ValueError(f"y_score length ({len(self.y_score)}) must equal y_pred length ({n})")
        return self
