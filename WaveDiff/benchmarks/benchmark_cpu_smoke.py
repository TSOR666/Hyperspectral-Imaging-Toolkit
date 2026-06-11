"""Small reproducible CPU microbenchmark for WaveDiff optimization paths."""

import json
import sys
import time
from pathlib import Path

import torch


WAVEDIFF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAVEDIFF_ROOT))

from models.base_model import HSILatentDiffusionModel
from modules.attention import CrossSpectralAttention


def _measure(function, warmup=1, iterations=5):
    for _ in range(warmup):
        function()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    attention_input = torch.randn(1, 64, 32, 32)
    attention_ms = {}
    for mode in ("spatial", "windowed", "channel"):
        module = CrossSpectralAttention(
            64,
            mode=mode,
            window_size=8,
        ).eval()
        attention_ms[mode] = _measure(
            lambda: module(attention_input),
            iterations=10,
        )

    legacy_model = HSILatentDiffusionModel(
        latent_dim=16,
        timesteps=8,
        norm_type="batch",
        cross_attention_mode="spatial",
        conditional_residual_diffusion=False,
        use_enhanced_attention=False,
        use_domain_adaptation=False,
    ).eval()
    legacy_parameter_count = sum(
        parameter.numel() for parameter in legacy_model.parameters()
    )

    model = HSILatentDiffusionModel(
        latent_dim=16,
        timesteps=8,
        norm_type="group",
        cross_attention_mode="channel",
        conditional_residual_diffusion=True,
        use_enhanced_attention=False,
        use_domain_adaptation=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    rgb1 = torch.randn(1, 3, 32, 32)
    rgb4 = torch.randn(4, 3, 32, 32)
    with torch.inference_mode():
        legacy_batch1_ms = _measure(
            lambda: legacy_model.rgb_to_hsi(rgb1),
            iterations=5,
        )
        batch1_ms = _measure(lambda: model.rgb_to_hsi(rgb1), iterations=5)
        batch4_ms = _measure(lambda: model.rgb_to_hsi(rgb4), iterations=5)

    hsi = torch.randn(1, 31, 32, 32)
    legacy_model.train()
    legacy_optimizer = torch.optim.AdamW(legacy_model.parameters(), lr=1e-4)

    def legacy_train_step():
        legacy_optimizer.zero_grad(set_to_none=True)
        outputs = legacy_model(rgb1)
        losses = legacy_model.calculate_losses(outputs, rgb1, hsi)
        total = (
            losses["diffusion_loss"]
            + losses["cycle_loss"]
            + losses["l1_loss"]
        )
        total.backward()
        legacy_optimizer.step()

    legacy_train_ms = _measure(
        legacy_train_step,
        warmup=1,
        iterations=3,
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        outputs = model(rgb1, hsi_target=hsi)
        losses = model.calculate_losses(outputs, rgb1, hsi)
        total = (
            losses["diffusion_loss"]
            + losses["cycle_loss"]
            + losses["l1_loss"]
            + 0.5 * losses["latent_reconstruction_loss"]
        )
        total.backward()
        optimizer.step()

    train_ms = _measure(train_step, warmup=1, iterations=3)

    print(json.dumps({
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "attention_ms_32x32": attention_ms,
        "channel_vs_spatial_speedup": (
            attention_ms["spatial"] / attention_ms["channel"]
        ),
        "windowed_vs_spatial_speedup": (
            attention_ms["spatial"] / attention_ms["windowed"]
        ),
        "legacy_model_parameters": legacy_parameter_count,
        "conditioned_model_parameters": parameter_count,
        "parameter_overhead_percent": (
            (parameter_count / legacy_parameter_count - 1.0) * 100.0
        ),
        "legacy_direct_inference_batch1_ms": legacy_batch1_ms,
        "direct_inference_batch1_ms": batch1_ms,
        "direct_inference_batch4_ms": batch4_ms,
        "batch1_images_per_second": 1000.0 / batch1_ms,
        "batch4_images_per_second": 4000.0 / batch4_ms,
        "legacy_train_step_ms": legacy_train_ms,
        "conditioned_train_step_ms": train_ms,
        "conditioned_train_overhead_percent": (
            (train_ms / legacy_train_ms - 1.0) * 100.0
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
