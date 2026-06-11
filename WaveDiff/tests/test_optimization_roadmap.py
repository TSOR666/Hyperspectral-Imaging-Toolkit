"""Regression tests for the WaveDiff optimization roadmap."""

import os
import sys

import numpy as np
import torch
from PIL import Image


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockMatplotlib:
    def __getattr__(self, name):
        return MockMatplotlib()

    def __call__(self, *args, **kwargs):
        return MockMatplotlib()

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


sys.modules.setdefault("matplotlib", MockMatplotlib())
sys.modules.setdefault("matplotlib.pyplot", MockMatplotlib())
sys.modules.setdefault("matplotlib.colors", MockMatplotlib())
sys.modules.setdefault("matplotlib.cm", MockMatplotlib())


def test_npy_dataset_cache_avoids_redecode(tmp_path, monkeypatch):
    from train import HSIDataset

    rgb_dir = tmp_path / "RGB"
    hsi_dir = tmp_path / "HSI"
    rgb_dir.mkdir()
    hsi_dir.mkdir()
    Image.new("RGB", (8, 8)).save(rgb_dir / "sample.png")
    np.save(hsi_dir / "sample.npy", np.zeros((8, 8, 31), np.float32))

    calls = 0
    transform_calls = 0
    original_load = np.load

    def counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)

    def hsi_transform(tensor):
        nonlocal transform_calls
        transform_calls += 1
        return tensor + 1.0

    dataset = HSIDataset(
        tmp_path,
        transform=lambda image: torch.zeros(3, 8, 8),
        hsi_transform=hsi_transform,
        npy_cache_size=1,
    )

    first = dataset[0]["hsi"]
    second = dataset[0]["hsi"]

    assert calls == 1
    assert transform_calls == 1
    assert dataset.cache_hits == 1
    assert first.data_ptr() != second.data_ptr()


def test_cross_attention_modes_preserve_shape():
    from modules.attention import CrossSpectralAttention

    x = torch.randn(2, 16, 9, 11)
    for mode in ("channel", "windowed"):
        module = CrossSpectralAttention(
            16,
            num_heads=4,
            mode=mode,
            window_size=4,
        )
        output = module(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()


def test_group_norm_is_batch_size_independent():
    from modules.encoders import ResidualBlock

    block = ResidualBlock(12, norm_type="group", norm_groups=8)
    assert isinstance(block.bn1, torch.nn.GroupNorm)
    output = block(torch.randn(1, 12, 8, 8))
    assert output.shape == (1, 12, 8, 8)


def test_conditioned_residual_diffusion_train_and_sample():
    from models.base_model import HSILatentDiffusionModel

    model = HSILatentDiffusionModel(
        latent_dim=16,
        timesteps=8,
        use_batchnorm=False,
        norm_type="none",
        cross_attention_mode="channel",
        conditional_residual_diffusion=True,
    )
    rgb = torch.randn(1, 3, 16, 16)
    hsi = torch.randn(1, 31, 16, 16)
    outputs = model(rgb, hsi_target=hsi)

    assert outputs["target_latent"] is not None
    assert outputs["target_latent_hsi"].shape == hsi.shape
    assert torch.isfinite(outputs["diffusion_loss"])

    sampled = model.rgb_to_hsi(
        rgb,
        latent_mode="diffusion",
        sampling_steps=2,
    )
    assert sampled.shape == hsi.shape
    assert torch.isfinite(sampled).all()


def test_tiled_inference_stitches_pixelwise_model_exactly():
    from inference import run_tiled_inference

    class PixelwiseModel(torch.nn.Module):
        def rgb_to_hsi(self, rgb, **_):
            output = rgb[:, :1].repeat(1, 31, 1, 1)
            return output, {"initial": output, "final": output}

    rgb = torch.randn(2, 3, 20, 24)
    output, stages = run_tiled_inference(
        PixelwiseModel(),
        rgb,
        torch.device("cpu"),
        tile_size=12,
        overlap=4,
        tile_batch_size=3,
    )
    expected = rgb[:, :1].repeat(1, 31, 1, 1)

    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.allclose(stages["initial"], expected, atol=1e-6)


def test_directory_inference_uses_requested_batch_size(tmp_path, monkeypatch):
    import inference

    input_dir = tmp_path / "RGB"
    output_dir = tmp_path / "results"
    input_dir.mkdir()
    for index in range(3):
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(
            input_dir / f"{index}.png"
        )

    batch_sizes = []

    def fake_run(model, rgb, device, **kwargs):
        batch_sizes.append(rgb.shape[0])
        output = torch.zeros(rgb.shape[0], 31, 8, 8)
        return output, {"final": output}

    monkeypatch.setattr(inference, "run_inference", fake_run)
    monkeypatch.setattr(inference, "save_results", lambda *args, **kwargs: None)
    inference.process_directory(
        input_dir,
        object(),
        torch.device("cpu"),
        output_dir,
        batch_size=2,
        config={"image_size": 8},
    )

    assert batch_sizes == [2, 1]


def test_ema_checkpoint_state_round_trip():
    from utils.ema import ModelEMA

    model = torch.nn.Conv2d(3, 4, 1)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)

    restored = ModelEMA(model, decay=0.5)
    restored.load_state_dict(ema.state_dict())

    assert restored.decay == 0.9
    assert restored.num_updates == 1
    for expected, actual in zip(
        ema.module.parameters(),
        restored.module.parameters(),
    ):
        assert torch.allclose(expected, actual)


def test_legacy_checkpoint_initializes_ema_from_loaded_model(tmp_path):
    from train import load_checkpoint, save_checkpoint
    from utils.ema import ModelEMA

    source = torch.nn.Conv2d(3, 4, 1)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: 1.0,
    )
    path = tmp_path / "legacy.pt"
    save_checkpoint(
        source,
        optimizer,
        scheduler,
        epoch=0,
        train_loss=1.0,
        val_loss=1.0,
        config={},
        path=path,
    )

    restored = torch.nn.Conv2d(3, 4, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda _: 1.0,
    )
    ema = ModelEMA(restored)
    load_checkpoint(
        restored,
        restored_optimizer,
        restored_scheduler,
        path,
        torch.device("cpu"),
        ema=ema,
    )

    for model_parameter, ema_parameter in zip(
        restored.parameters(),
        ema.module.parameters(),
    ):
        assert torch.allclose(model_parameter, ema_parameter)
