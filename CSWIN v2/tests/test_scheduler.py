import pytest
import torch

from hsi_model.training_script_fixed import WarmupCosineScheduler


def test_warmup_cosine_scheduler_uses_absolute_eta_min():
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=10,
        eta_min=1e-5,
    )

    scheduler.last_epoch = 10
    assert scheduler.get_lr()[0] == pytest.approx(1e-5, rel=1e-6)
