from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from hsifusion_training import HSIFusionTrainer, should_optimizer_step as hsifusion_should_step
from hsifusion_classifier_v253 import create_hsifusion_lightning_classifier
from hsifusion_v252_complete import RobustEnhancedSpectralAttention, create_hsifusion_lightning_pro
from optimized_dataloader import MSTPlusPlusLoss, create_optimized_dataloaders
from sharp_training_script_fixed import (
    DedicatedSHARPTrainer,
    SHARPTrainingConfig,
    should_optimizer_step as sharp_should_step,
)
from sharp_config_loader import parse_config_value
from sharp_inference import SHARPInference, _torch_load_compat
from sharp_v322_hardened import (
    SHARPv32,
    SHARPv32Config,
    SHARPv32Trainer,
    create_sharp_v32,
)


def _local_tmp_dir(name: str) -> Path:
    path = Path(".tmp_pytest_local") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_sharp_factory_forward_smoke() -> None:
    model = create_sharp_v32(
        model_size="tiny",
        in_channels=3,
        out_channels=31,
        compile_model=False,
        verbose=False,
        sparse_max_tokens=512,
        sparse_block_size=64,
        sparse_q_block_size=64,
        sparse_window_size=7,
        sparse_sparsity_ratio=0.5,
        sparse_k_cap=32,
    )
    model.eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 31, 16, 16)
    assert torch.isfinite(y).all()


def test_sharp_trainer_evaluate_returns_contract() -> None:
    cfg = SHARPv32Config(
        base_dim=16,
        depths=(1, 1, 1, 1),
        heads=(1, 2, 4, 8),
        mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        sparse_max_tokens=512,
        sparse_block_size=64,
        sparse_q_block_size=64,
        sparse_window_size=7,
        sparse_sparsity_ratio=0.5,
        sparse_k_cap=32,
    )
    model = SHARPv32(cfg)
    trainer = SHARPv32Trainer(model=model, total_steps=4, use_amp=False)

    x = torch.rand(1, 3, 16, 16)
    y = torch.zeros(1, 31, 16, 16)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    metrics = trainer.evaluate(loader)

    for key in ("loss", "mrae", "rmse", "psnr"):
        assert key in metrics
        assert math.isfinite(float(metrics[key]))


def test_mstplusplus_loss_div_zero_safe() -> None:
    criterion = MSTPlusPlusLoss(eps=1e-6)
    pred = torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    loss = criterion(pred, target)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_hsifusion_compute_metrics_div_zero_safe() -> None:
    trainer = HSIFusionTrainer.__new__(HSIFusionTrainer)
    trainer.config = SimpleNamespace(min_mrae_denom=1e-6)
    pred = torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    metrics = HSIFusionTrainer._compute_metrics(trainer, pred, target)
    for key in ("mrae", "rmse", "psnr"):
        assert math.isfinite(float(metrics[key]))


def test_hsifusion_low_rank_spectral_attention_forward() -> None:
    dense = RobustEnhancedSpectralAttention(dim=64, num_bands=31, pool_sizes=[2])
    low_rank = RobustEnhancedSpectralAttention(
        dim=64,
        num_bands=31,
        pool_sizes=[2],
        spectral_basis_rank=8,
    )

    assert dense.spectral_weights is not None
    assert dense.spectral_weights[0].numel() == 31 * 31
    assert low_rank.spectral_coeffs is not None
    assert low_rank.spectral_coeffs[0].numel() == 8

    x = torch.rand(1, 64, 16, 16)
    with torch.no_grad():
        y = low_rank(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_optimizer_step_remainder_batch() -> None:
    # 3 batches, accumulate every 2: last batch must still trigger an optimizer step.
    assert hsifusion_should_step(batch_idx=1, total_batches=3, accumulate_steps=2)
    assert hsifusion_should_step(batch_idx=2, total_batches=3, accumulate_steps=2)
    assert sharp_should_step(batch_idx=2, total_batches=3, accumulate_steps=2)


def test_sharp_train_step_skips_nonfinite_loss() -> None:
    cfg = SHARPv32Config(
        base_dim=16,
        depths=(1, 1, 1, 1),
        heads=(1, 2, 4, 8),
        mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        sparse_max_tokens=512,
        sparse_block_size=64,
        sparse_q_block_size=64,
        sparse_window_size=7,
        sparse_sparsity_ratio=0.5,
        sparse_k_cap=32,
    )
    model = SHARPv32(cfg)
    trainer = SHARPv32Trainer(model=model, total_steps=4, use_amp=False)

    def _nan_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), device=targets.device)

    trainer.model.compute_loss = _nan_loss  # type: ignore[assignment]
    metrics = trainer.train_step(torch.rand(1, 3, 16, 16), torch.rand(1, 31, 16, 16))
    assert metrics["skipped"] is True


def test_torch_load_compat_handles_non_weights_payload() -> None:
    checkpoint_path = _local_tmp_dir("torch_load_compat") / "ckpt.pth"
    payload = {"config": SimpleNamespace(model_size="tiny", in_channels=3, out_channels=31)}
    torch.save(payload, checkpoint_path)
    loaded = _torch_load_compat(str(checkpoint_path), torch.device("cpu"))
    assert isinstance(loaded["config"], SimpleNamespace)
    assert loaded["config"].model_size == "tiny"


def test_process_image_file_no_unboundlocalerror() -> None:
    tmp_dir = _local_tmp_dir("process_image_file")
    rgb = (np.random.rand(8, 8, 3) * 255.0).astype(np.uint8)
    image_path = tmp_dir / "rgb.png"
    output_path = tmp_dir / "pred.npy"
    Image.fromarray(rgb).save(image_path)

    runner = SHARPInference.__new__(SHARPInference)
    runner.device = torch.device("cpu")
    runner.out_channels = 31

    def _predict(rgb_image: torch.Tensor, patch_size=None, overlap: int = 16) -> torch.Tensor:
        if rgb_image.ndim == 3:
            _, h, w = rgb_image.shape
        else:
            _, _, h, w = rgb_image.shape
        return torch.ones((31, h, w), dtype=torch.float32)

    runner.predict = _predict
    out = SHARPInference.process_image_file(runner, str(image_path), output_path=str(output_path))
    assert out.shape == (31, 8, 8)
    assert output_path.exists()


def test_blend_weight_has_positive_floor() -> None:
    runner = SHARPInference.__new__(SHARPInference)
    weight = SHARPInference._create_blend_weight(runner, 16, 16)
    assert float(weight.min()) > 0.0


def test_sharp_inference_rejects_channel_mismatch() -> None:
    runner = SHARPInference.__new__(SHARPInference)
    runner.device = torch.device("cpu")
    runner.in_channels = 3
    runner.out_channels = 31
    runner.model = nn.Identity()

    with pytest.raises(ValueError, match="Expected 3 channels"):
        SHARPInference.predict(runner, torch.rand(1, 4, 8, 8))


def test_optimized_dataloaders_respect_augment_flag_and_png_inputs() -> None:
    tmp_dir = _local_tmp_dir("png_dataset")
    (tmp_dir / "split_txt").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "Train_RGB").mkdir(exist_ok=True)
    (tmp_dir / "Train_Spec").mkdir(exist_ok=True)

    rgb = (np.random.rand(8, 8, 3) * 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_dir / "Train_RGB" / "sample.png")
    with h5py.File(tmp_dir / "Train_Spec" / "sample.mat", "w") as handle:
        handle.create_dataset("cube", data=np.random.rand(31, 8, 8).astype(np.float32))

    (tmp_dir / "split_txt" / "train_list.txt").write_text("sample\n", encoding="utf-8")
    (tmp_dir / "split_txt" / "valid_list.txt").write_text("sample\n", encoding="utf-8")

    config = SimpleNamespace(
        data_root=str(tmp_dir),
        batch_size=1,
        num_workers=0,
        val_num_workers=0,
        memory_mode="standard",
        patch_size=4,
        stride=4,
        augment=False,
        seed=7,
    )
    train_loader, val_loader = create_optimized_dataloaders(config)

    assert train_loader.dataset.augment is False
    assert train_loader.dataset.rgb_files[0].endswith(".png")
    rgb_batch, hsi_batch = next(iter(train_loader))
    assert rgb_batch.shape == (1, 3, 4, 4)
    assert hsi_batch.shape == (1, 31, 4, 4)
    assert len(val_loader) == 1


def test_dedicated_sharp_trainer_forwards_sparse_config_to_factory(monkeypatch) -> None:
    captured = {}

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        @property
        def num_parameters(self):
            return {"total": 1, "trainable": 1, "size_mb": 0.0}

    def _fake_factory(*args, **kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr("sharp_training_script_fixed.create_sharp_v32", _fake_factory)

    trainer = DedicatedSHARPTrainer.__new__(DedicatedSHARPTrainer)
    trainer.config = SHARPTrainingConfig(
        model_size="tiny",
        compile_model=False,
        use_checkpoint=True,
        sparse_sparsity_ratio=0.35,
        sparse_block_size=321,
        sparse_q_block_size=123,
        sparse_window_size=17,
        sparse_max_tokens=777,
        sparse_k_cap=55,
        rbf_centers_per_head=11,
        key_rbf_mode="linear",
        ema_update_every=5,
        device="cpu",
    )
    trainer.device = torch.device("cpu")

    model = DedicatedSHARPTrainer._create_model(trainer)

    assert isinstance(model, DummyModel)
    assert captured["use_checkpoint"] is True
    assert captured["sparse_block_size"] == 321
    assert captured["sparse_q_block_size"] == 123
    assert captured["sparse_window_size"] == 17
    assert captured["sparse_max_tokens"] == 777
    assert captured["sparse_k_cap"] == 55
    assert captured["rbf_centers_per_head"] == 11
    assert captured["key_rbf_mode"] == "linear"
    assert captured["ema_update_every"] == 5


def test_sharp_trainer_psnr_defaults_to_unit_range() -> None:
    class DummyEvalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(ema_update_every=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 31, x.shape[2], x.shape[3]), device=x.device)

        def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean((pred - target) ** 2)

    trainer = SHARPv32Trainer(model=DummyEvalModel(), total_steps=4, use_amp=False)
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 3, 4, 4), torch.ones(1, 31, 4, 4)),
        batch_size=1,
    )
    metrics = trainer.evaluate(loader)

    assert abs(metrics["psnr"]) < 1e-6


def test_sharp_config_loader_parses_scientific_notation() -> None:
    assert parse_config_value("4e-4") == pytest.approx(4e-4)
    assert parse_config_value("1E-3") == pytest.approx(1e-3)


def test_hsifusion_factory_explicit_none_disables_compile(monkeypatch) -> None:
    compile_calls = []

    def _compile_should_not_run(model, **kwargs):
        compile_calls.append(kwargs)
        raise AssertionError("torch.compile should not be called when compile_mode=None")

    monkeypatch.setattr(torch, "compile", _compile_should_not_run)
    model = create_hsifusion_lightning_pro(
        model_size="tiny",
        in_channels=3,
        out_channels=31,
        compile_mode=None,
        expected_min_size=256,
        skip_compile_small_inputs=False,
    )
    assert compile_calls == []
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    assert y.shape == (1, 31, 64, 64)


def test_hsifusion_classifier_explicit_none_disables_compile(monkeypatch) -> None:
    compile_calls = []

    def _compile_should_not_run(model, **kwargs):
        compile_calls.append(kwargs)
        raise AssertionError("torch.compile should not be called when compile_mode=None")

    monkeypatch.setattr(torch, "compile", _compile_should_not_run)
    model = create_hsifusion_lightning_classifier(
        model_size="tiny",
        in_channels=3,
        num_classes=7,
        compile_mode=None,
        expected_min_size=256,
        skip_compile_small_inputs=False,
    )
    assert compile_calls == []
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 7)
