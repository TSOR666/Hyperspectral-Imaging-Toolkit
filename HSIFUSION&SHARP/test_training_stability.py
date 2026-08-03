"""Regression tests for mixed-precision MRAE and checkpoint stability."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from optimized_dataloader import MSTPlusPlusLoss, _validate_hsi_is_finite
from training_stability import (
    annealed_mrae_epsilon,
    make_grad_scaler,
    resolve_amp_dtype,
)


def test_mrae_uses_fp32_for_half_precision_dark_targets() -> None:
    prediction = torch.ones(1, 31, 4, 4, dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(prediction)

    loss = MSTPlusPlusLoss(eps=1e-6)(prediction, target)
    loss.backward()

    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(1e6, rel=1e-3)
    assert torch.isfinite(prediction.grad).all()


def test_annealed_mrae_epsilon_has_geometric_midpoint() -> None:
    assert annealed_mrae_epsilon(1e-2, 1e-4, 0, 100) == pytest.approx(1e-2)
    assert annealed_mrae_epsilon(1e-2, 1e-4, 50, 100) == pytest.approx(1e-3)
    assert annealed_mrae_epsilon(1e-2, 1e-4, 100, 100) == pytest.approx(1e-4)
    assert annealed_mrae_epsilon(1e-2, 1e-4, 200, 100) == pytest.approx(1e-4)


def test_amp_auto_prefers_bf16_and_falls_back_to_fp16(monkeypatch) -> None:
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp_dtype("auto", cuda) == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_amp_dtype("auto", cuda) == torch.float16
    assert resolve_amp_dtype("off", cuda) is None


def test_nonfinite_hsi_is_rejected_with_source_path() -> None:
    cube = np.zeros((31, 2, 2), dtype=np.float32)
    cube[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=r"bad\.mat.*NaN/Inf"):
        _validate_hsi_is_finite(cube, "bad.mat")


def _fake_unified_trainer(epochs: int, val_interval: int):
    from unified_training import UnifiedTrainer

    trainer = object.__new__(UnifiedTrainer)
    trainer.config = SimpleNamespace(epochs=epochs, val_interval=val_interval)
    trainer.start_epoch = 0
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 4e-4}])
    trainer.best_mrae = float("inf")
    trainer.writer = None
    return trainer


def test_unified_trainer_writes_last_checkpoint_each_epoch() -> None:
    trainer = _fake_unified_trainer(epochs=3, val_interval=2)
    trainer._train_epoch = lambda epoch: float(epoch + 1)
    trainer.validate = lambda epoch: {"full/mrae": 1.0 / epoch}
    saves = []
    trainer._save_checkpoint = lambda epoch, is_best: saves.append((epoch, is_best))

    trainer.train()

    assert saves == [(1, False), (2, True), (3, True)]


def test_unified_trainer_writes_recovery_checkpoint_on_first_epoch_failure() -> None:
    trainer = _fake_unified_trainer(epochs=2, val_interval=2)

    def fail(_epoch):
        raise RuntimeError("non-finite gradients")

    trainer._train_epoch = fail
    saves = []
    trainer._save_checkpoint = lambda epoch, is_best: saves.append((epoch, is_best))

    with pytest.raises(RuntimeError, match="non-finite gradients"):
        trainer.train()

    assert saves == [(0, False)]


def test_unified_trainer_aborts_on_persistent_nonfinite_gradients() -> None:
    """Finite losses must not reset the failure counter before backward succeeds."""
    from unified_training import UnifiedTrainer

    model = torch.nn.Conv2d(3, 31, kernel_size=1)
    model.weight.register_hook(
        lambda grad: torch.full_like(grad, float("nan"))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    trainer = object.__new__(UnifiedTrainer)
    trainer.config = SimpleNamespace(
        accumulate_steps=1,
        gradient_clip=1.0,
        max_consecutive_nonfinite=3,
        log_interval=100,
        checkpoint_interval_steps=0,
        train_mrae_eps_start=1e-2,
        train_mrae_eps_end=1e-3,
        train_mrae_eps_anneal_steps=100,
    )
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer._orig_model = model
    trainer.optimizer = optimizer
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    trainer.scaler = make_grad_scaler(trainer.device, None)
    trainer.criterion = MSTPlusPlusLoss(eps=1e-2)
    trainer.amp_dtype = None
    trainer.use_amp = False
    trainer.ema_state = None
    trainer.optimizer_step = 0
    trainer.iteration = 0
    trainer.consecutive_nonfinite = 0
    trainer.writer = None
    trainer.train_loader = [
        (torch.rand(1, 3, 2, 2), torch.rand(1, 31, 2, 2).clamp_min(0.1))
        for _ in range(3)
    ]

    with pytest.raises(RuntimeError, match="consecutive non-finite grads"):
        trainer._train_epoch(0)

    assert trainer.consecutive_nonfinite == 3
    assert trainer.optimizer_step == 0
