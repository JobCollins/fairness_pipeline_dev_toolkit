import torch

from fairness_pipeline_dev_toolkit.training import FairnessRegularizerLoss


def test_fairness_regularizer_covariance_mode_runs():
    N = 64
    logits = torch.randn(N)
    y = torch.randint(0, 2, (N,))
    s = torch.randint(0, 2, (N,))
    loss_fn = FairnessRegularizerLoss(eta=0.5, mode="covariance")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True


def test_fairness_regularizer_mean_gap_mode_runs():
    N = 64
    logits = torch.randn(N)
    y = torch.randint(0, 2, (N,))
    s = torch.randint(0, 2, (N,))
    loss_fn = FairnessRegularizerLoss(eta=1.0, mode="mean_gap")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True
