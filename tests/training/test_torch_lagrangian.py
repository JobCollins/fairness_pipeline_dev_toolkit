import torch
import torch.nn as nn

from fairness_pipeline_dev_toolkit.training import LagrangianFairnessTrainer


class TinyNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x)


def test_lagrangian_trainer_runs_dp():
    torch.manual_seed(0)
    N, d = 128, 5
    X = torch.randn(N, d)
    s = (torch.rand(N) > 0.5).long()
    # y depends on x0 and (undesirably) s
    y = (X[:, 0] + 0.4 * s + 0.1 * torch.randn(N) > 0.0).long()

    model = TinyNet(d)
    trainer = LagrangianFairnessTrainer(
        model=model,
        fairness="demographic_parity",
        dp_tolerance=0.2,  # fairly loose to stabilize tiny example
        model_lr=5e-3,
        lambda_lr=1e-2,
        device="cpu",
    )
    hist = trainer.fit(X, y, s, epochs=3, batch_size=64, verbose=False)
    assert isinstance(hist, list) and len(hist) > 0
    assert {"epoch", "accuracy", "violation", "lambda"}.issubset(hist[-1].keys())


def test_lagrangian_trainer_runs_eo():
    torch.manual_seed(0)
    N, d = 128, 5
    X = torch.randn(N, d)
    s = (torch.rand(N) > 0.5).long()
    y = (X[:, 0] + 0.4 * s + 0.1 * torch.randn(N) > 0.0).long()

    model = TinyNet(d)
    trainer = LagrangianFairnessTrainer(
        model=model,
        fairness="equal_opportunity",
        eo_tolerance=0.3,
        model_lr=5e-3,
        lambda_lr=1e-2,
        device="cpu",
    )
    hist = trainer.fit(X, y, s, epochs=3, batch_size=64, verbose=False)
    assert isinstance(hist, list) and len(hist) > 0
