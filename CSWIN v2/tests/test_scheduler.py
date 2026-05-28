import pytest
import torch

from hsi_model.training_script_fixed import (
    WarmupCosineScheduler,
    scheduler_steps_for_accumulation,
)
from hsi_model.train_optimized import resolve_generator_discriminator_lrs


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


def test_scheduler_steps_match_optimizer_updates_under_accumulation():
    assert scheduler_steps_for_accumulation(300_000, 2) == 150_000
    assert scheduler_steps_for_accumulation(5, 2) == 3
    assert scheduler_steps_for_accumulation(5, 0) == 5


def test_train_optimized_resolves_split_generator_discriminator_lrs():
    generator_lr, discriminator_lr = resolve_generator_discriminator_lrs(
        {
            "learning_rate": 4e-4,
            "generator_lr": 1e-4,
            "discriminator_lr": 2e-5,
        }
    )
    assert generator_lr == pytest.approx(1e-4)
    assert discriminator_lr == pytest.approx(2e-5)


def test_train_optimized_preserves_legacy_learning_rate_fallback():
    generator_lr, discriminator_lr = resolve_generator_discriminator_lrs(
        {"learning_rate": 4e-4}
    )
    assert generator_lr == pytest.approx(4e-4)
    assert discriminator_lr == pytest.approx(4e-4)
