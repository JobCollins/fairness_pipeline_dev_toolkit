"""
Training Module public API.

Re-exports convenience symbols so users can import like:
  from fairness_pipeline_dev_toolkit.training import (
      ReductionsWrapper,
      FairnessRegularizerLoss,
      LagrangianFairnessTrainer,
      GroupFairnessCalibrator,
      sweep_pareto,
      plot_pareto,
  )

Imports are lazy so that, for example, ``ReductionsWrapper`` can be resolved without
importing PyTorch-backed symbols (and vice versa).
"""

from __future__ import annotations

from typing import Any, List

__all__ = [
    "ReductionsWrapper",
    "FairnessRegularizerLoss",
    "LagrangianFairnessTrainer",
    "GroupFairnessCalibrator",
    "sweep_pareto",
    "plot_pareto",
]


def __getattr__(name: str) -> Any:
    if name == "ReductionsWrapper":
        from .sklearn_.reductions_wrapper import ReductionsWrapper

        return ReductionsWrapper
    if name == "GroupFairnessCalibrator":
        from .postproc.calibration import GroupFairnessCalibrator

        return GroupFairnessCalibrator
    if name == "LagrangianFairnessTrainer":
        from .torch_.lagrangian import LagrangianFairnessTrainer

        return LagrangianFairnessTrainer
    if name == "FairnessRegularizerLoss":
        from .torch_.losses import FairnessRegularizerLoss

        return FairnessRegularizerLoss
    if name == "sweep_pareto":
        from .viz.pareto import sweep_pareto

        return sweep_pareto
    if name == "plot_pareto":
        from .viz.pareto import plot_pareto

        return plot_pareto
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
