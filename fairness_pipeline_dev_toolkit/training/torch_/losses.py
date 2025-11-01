from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


class FairnessRegularizerLoss(nn.Module):
    """
    BCEWithLogits + fairness penalty.

    The fairness term penalizes dependence between predictions and a binary
    sensitive attribute using a differentiable proxy (covariance of probs and S)
    or mean-gap squared between groups.

    Parameters
    ----------
    eta : float
        Strength of the fairness penalty.
    mode : {"covariance", "mean_gap"}
        - "covariance": penalize Cov(sigmoid(logits), S)
        - "mean_gap": penalize (mean_pred[S=0] - mean_pred[S=1])^2
    pos_weight : Optional[float]
        Optional positive class weight forwarded to BCEWithLogitsLoss.

    Notes
    -----
    - Aligns with Training Module requirement #2.
    - Binary sensitive attribute expected: S ∈ {0,1}.
    """

    def __init__(
        self,
        eta: float = 0.5,
        mode: Literal["covariance", "mean_gap"] = "covariance",
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.eta = float(eta)
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight) if pos_weight is not None else None
        )

    def forward(
        self, logits: torch.Tensor, y_true: torch.Tensor, s_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args
        ----
        logits: (N,) or (N,1) raw model outputs
        y_true: (N,) binary labels {0,1}
        s_attr: (N,) binary sensitive attribute {0,1}

        Returns
        -------
        total_loss: torch.Tensor
        """
        logits = logits.view(-1)
        y_true = y_true.view(-1).float()
        s_attr = s_attr.view(-1).float()

        # 1) accuracy term
        acc_loss = self.bce(logits, y_true)

        # 2) fairness term (probabilities)
        probs = torch.sigmoid(logits)
        if self.mode == "covariance":
            p_s1 = s_attr.mean()
            p_y1 = probs.mean()
            p_y1_s1 = (probs * s_attr).mean()
            fairness = torch.abs(p_y1_s1 - p_y1 * p_s1)  # |Cov|
        elif self.mode == "mean_gap":
            # Avoid zero-div; if any group absent, fairness = 0
            mask0 = s_attr == 0
            mask1 = s_attr == 1
            if mask0.any() and mask1.any():
                m0 = probs[mask0].mean()
                m1 = probs[mask1].mean()
                fairness = (m0 - m1) ** 2
            else:
                fairness = torch.zeros((), device=logits.device)
        else:
            raise ValueError("Unknown mode for fairness regularizer.")

        return acc_loss + self.eta * fairness
