from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

import torch
import torch.nn as nn
import torch.optim as optim

ConstraintKind = Literal["demographic_parity", "equal_opportunity"]


@dataclass
class LagrangianFairnessTrainer:
    """
    Lagrangian trainer enforcing a fairness constraint with a dual variable (lambda).

    Parameters
    ----------
    model : nn.Module
        Binary classifier producing logits.
    fairness : {"demographic_parity", "equal_opportunity"}
        Constraint to enforce.
    dp_tolerance : float
        Allowed difference between groups (S=0 vs S=1).
    eo_tolerance : float
        Allowed TPR difference for Equal Opportunity.
    model_lr : float
        Learning rate for model parameters (primal).
    lambda_lr : float
        Step size for dual ascent on the constraint violation.
    device : str
        "cpu" or "cuda".
    """

    model: nn.Module
    fairness: ConstraintKind = "demographic_parity"
    dp_tolerance: float = 0.02
    eo_tolerance: float = 0.02
    model_lr: float = 1e-3
    lambda_lr: float = 1e-2
    device: str = "cpu"

    # internal state: scalar dual variable (kept as Python float; no autograd)
    lambda_param: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.model.to(self.device)
        self._opt_model = optim.Adam(self.model.parameters(), lr=self.model_lr)

    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _dp_violation(self, logits: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Demographic parity violation surrogate:
            max(0, |P(Ŷ=1|S=1) - P(Ŷ=1|S=0)| - tol)
        Returns unclamped (gap - tol); caller decides clamping.
        """
        probs = self._sigmoid(logits)
        s = s.view(-1).float()
        m0 = probs[s == 0].mean() if (s == 0).any() else torch.tensor(0.0, device=probs.device)
        m1 = probs[s == 1].mean() if (s == 1).any() else torch.tensor(0.0, device=probs.device)
        gap = torch.abs(m1 - m0)
        return gap - self.dp_tolerance

    def _eo_violation(self, logits: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Equal opportunity violation surrogate:
            max(0, |TPR(S=1) - TPR(S=0)| - tol)
        Returns unclamped (gap - tol); caller decides clamping.
        """
        probs = self._sigmoid(logits)
        y = y.view(-1).float()
        s = s.view(-1).float()
        yhat = (probs >= 0.5).float()

        def tpr(mask_s: torch.Tensor) -> torch.Tensor:
            mask_pos = (y == 1) & mask_s
            if mask_pos.any():
                return (yhat[mask_pos] == 1).float().mean()
            return torch.tensor(0.0, device=probs.device)

        tpr0 = tpr(s == 0)
        tpr1 = tpr(s == 1)
        gap = torch.abs(tpr1 - tpr0)
        return gap - self.eo_tolerance

    def _violation(self, logits: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.fairness == "demographic_parity":
            return self._dp_violation(logits, s)
        elif self.fairness == "equal_opportunity":
            return self._eo_violation(logits, y, s)
        raise ValueError("Unknown fairness constraint.")

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Train loop with alternating updates:
        - primal: minimize accuracy_loss + lambda * violation_surrogate
        - dual:   projected ascent   lambda <- max(0, lambda + lr * violation_surrogate)

        Returns a history list with metrics per epoch.
        """
        self.model.train()
        X = X.to(self.device)
        y = y.to(self.device).long()
        s = s.to(self.device).long()

        bce = nn.BCEWithLogitsLoss()
        history: List[Dict[str, float]] = []

        n = X.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            Xp, yp, sp = X[perm], y[perm], s[perm]

            for i in range(0, n, batch_size):
                xb = Xp[i : i + batch_size]
                yb = yp[i : i + batch_size]
                sb = sp[i : i + batch_size]

                logits = self.model(xb).view(-1)
                acc_loss = bce(logits, yb.float())

                # Unclamped violation (can be negative when within tolerance)
                viol = self._violation(logits, yb, sb)
                viol_pos = torch.clamp(viol, min=0.0)  # only penalize violations

                # ---- primal step: minimize acc_loss + lambda * viol_pos ----
                loss = acc_loss + float(self.lambda_param) * viol_pos
                self._opt_model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self._opt_model.step()

                # ---- dual step: projected ascent on violation ----
                # λ ← max(0, λ + η * violation)
                viol_value = float(viol_pos.detach().item())
                self.lambda_param = max(0.0, self.lambda_param + self.lambda_lr * viol_value)

            # epoch metrics
            with torch.no_grad():
                logits_all = self.model(X).view(-1)
                probs = torch.sigmoid(logits_all)
                pred = (probs >= 0.5).long()
                acc = (pred == y).float().mean().item()
                v = float(torch.clamp(self._violation(logits_all, y, s), min=0.0).item())
                history.append(
                    {
                        "epoch": ep,
                        "accuracy": acc,
                        "violation": v,
                        "lambda": float(self.lambda_param),
                    }
                )
                if verbose:
                    print(
                        f"[ep {ep:02d}] acc={acc:.3f} viol={v:.4f} lambda={self.lambda_param:.3f}"
                    )

        return history
