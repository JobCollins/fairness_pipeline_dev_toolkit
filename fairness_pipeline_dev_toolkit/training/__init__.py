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
"""

from .postproc.calibration import GroupFairnessCalibrator
from .sklearn_.reductions_wrapper import ReductionsWrapper
from .torch_.lagrangian import LagrangianFairnessTrainer
from .torch_.losses import FairnessRegularizerLoss
from .viz.pareto import plot_pareto, sweep_pareto

__all__ = [
    "ReductionsWrapper",
    "FairnessRegularizerLoss",
    "LagrangianFairnessTrainer",
    "GroupFairnessCalibrator",
    "sweep_pareto",
    "plot_pareto",
]
