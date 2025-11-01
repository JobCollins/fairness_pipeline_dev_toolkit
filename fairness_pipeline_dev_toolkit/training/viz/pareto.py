from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..torch_.losses import FairnessRegularizerLoss


def _demo_net(d_in: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _dp_diff_from_logits(logits: torch.Tensor, s: torch.Tensor) -> float:
    probs = torch.sigmoid(logits.view(-1))
    s = s.view(-1).float()
    m0 = probs[s == 0].mean() if (s == 0).any() else torch.tensor(0.0, device=probs.device)
    m1 = probs[s == 1].mean() if (s == 1).any() else torch.tensor(0.0, device=probs.device)
    return float(torch.abs(m1 - m0).item())


def sweep_pareto(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    s_val: np.ndarray,
    etas: Iterable[float] = (0.0, 0.1, 0.5, 1.0, 2.0),
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """
    Train simple NN with FairnessRegularizerLoss for multiple eta values
    and report accuracy and demographic parity difference.

    Returns a list of dicts: [{"eta": ..., "accuracy": ..., "dp_diff": ...}, ...]

    Notes
    -----
    - Aligns with Training Module requirement #5 (Pareto frontier).
    """
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    str_ = torch.tensor(s_train, dtype=torch.long, device=device)

    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    sv = torch.tensor(s_val, dtype=torch.long, device=device)

    d_in = Xtr.shape[1]
    out: List[Dict[str, float]] = []

    for eta in etas:
        model = _demo_net(d_in).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = FairnessRegularizerLoss(eta=float(eta), mode="covariance")

        # train
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(Xtr).view(-1)
            loss = crit(logits, ytr, str_)
            loss.backward()
            opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            logits_v = model(Xv).view(-1)
            probs_v = torch.sigmoid(logits_v)
            pred_v = (probs_v >= 0.5).long()
            acc = float((pred_v == yv).float().mean().item())
            dpd = _dp_diff_from_logits(logits_v, sv)

        out.append({"eta": float(eta), "accuracy": acc, "dp_diff": dpd})

    return out


def plot_pareto(points: List[Dict[str, float]], save_path: Optional[str] = None):
    """
    Plot Accuracy (y-axis) vs DP Difference (x-axis).
    """
    import matplotlib.pyplot as plt  # lazy import to avoid hard dep at import time

    xs = [p["dp_diff"] for p in points]
    ys = [p["accuracy"] for p in points]
    labels = [p["eta"] for p in points]

    plt.figure()
    plt.scatter(xs, ys)
    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(f"η={lbl}", (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Demographic Parity Difference (↓ better)")
    plt.ylabel("Accuracy (↑ better)")
    plt.title("Pareto Frontier: Accuracy vs Fairness")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    else:
        plt.show()
