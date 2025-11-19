import pytest
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


def test_fairness_regularizer_single_group_covariance():
    """Test covariance mode with only one group present."""
    N = 32
    logits = torch.randn(N)
    y = torch.randint(0, 2, (N,))
    s = torch.zeros(N)  # All same group
    loss_fn = FairnessRegularizerLoss(eta=0.5, mode="covariance")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True


def test_fairness_regularizer_single_group_mean_gap():
    """Test mean_gap mode with only one group present (should return 0 fairness penalty)."""
    N = 32
    logits = torch.randn(N)
    y = torch.randint(0, 2, (N,))
    s = torch.zeros(N)  # All same group
    loss_fn = FairnessRegularizerLoss(eta=1.0, mode="mean_gap")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True
    # Fairness term should be 0 when only one group
    assert loss.item() >= 0  # Should be non-negative (just accuracy loss)


def test_fairness_regularizer_all_same_label():
    """Test with all labels the same."""
    N = 32
    logits = torch.randn(N)
    y = torch.ones(N)  # All positive
    s = torch.randint(0, 2, (N,))
    loss_fn = FairnessRegularizerLoss(eta=0.5, mode="covariance")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True


def test_fairness_regularizer_eta_zero():
    """Test with eta=0 (no fairness penalty, just accuracy)."""
    N = 32
    logits = torch.randn(N)
    y = torch.randint(0, 2, (N,))
    s = torch.randint(0, 2, (N,))
    loss_fn = FairnessRegularizerLoss(eta=0.0, mode="covariance")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True
    assert loss.item() >= 0


def test_fairness_regularizer_large_eta():
    """Test with large eta value."""
    N = 32
    logits = torch.randn(N)
    y = torch.randint(0, 2, (N,))
    s = torch.randint(0, 2, (N,))
    loss_fn = FairnessRegularizerLoss(eta=10.0, mode="mean_gap")
    loss = loss_fn(logits, y, s)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True


def test_fairness_regularizer_invalid_mode():
    """Test that invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mode"):
        loss_fn = FairnessRegularizerLoss(eta=0.5, mode="invalid_mode")
        logits = torch.randn(10)
        y = torch.randint(0, 2, (10,))
        s = torch.randint(0, 2, (10,))
        loss_fn(logits, y, s)
