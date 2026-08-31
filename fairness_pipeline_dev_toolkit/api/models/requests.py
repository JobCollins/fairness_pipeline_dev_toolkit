from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fairness_pipeline_dev_toolkit.llm_evals.guards import DEFAULT_LLM_MIN_GROUP_SIZE


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


class LLMEvalRequest(BaseModel):
    """POST /llm-eval body: YAML text and/or JSON fields matching ``llm_eval:``.

    Extra keys (including credential field names) are kept so
    ``load_llm_eval_config()`` can reject them on one code path. REST-only
    knobs (``threshold``, ``metric``, ``min_group_size``, ``with_ci``) are
    not part of the YAML schema.
    """

    model_config = ConfigDict(extra="allow")

    config: Optional[str] = Field(
        default=None,
        description="YAML text of an llm_eval block (standalone or wrapped in llm_eval:).",
    )
    provider: Optional[str] = None
    model: Optional[str] = None
    evaluators: Optional[List[str]] = None
    counterfactual: Optional[Dict[str, Any]] = None
    prompt_templates: Optional[Dict[str, Any]] = None
    cache_dir: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    bbq_path: Optional[str] = None
    allow_small_samples: Optional[bool] = None
    max_requests_per_run: Optional[int] = None
    min_group_size: int = DEFAULT_LLM_MIN_GROUP_SIZE
    with_ci: bool = True
    ci_level: float = 0.95
    bootstrap_B: int = 200
    threshold: Optional[float] = None
    metric: Optional[str] = None
