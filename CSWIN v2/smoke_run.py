#!/usr/bin/env python3
"""Synthetic train/validation/inference smoke run for CSWIN v2."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hsi_model.models import (  # noqa: E402
    ComputeSinkhornDiscriminatorLoss,
    NoiseRobustCSWinModel,
    NoiseRobustLoss,
)
from hsi_model.utils.metrics import compute_metrics  # noqa: E402


def _config() -> dict[str, object]:
    return {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
        "lambda_rec": 1.0,
        "lambda_perceptual": 0.0,
        "lambda_adversarial": 0.1,
        "lambda_sam": 0.05,
        "sinkhorn_epsilon": 0.1,
        "sinkhorn_iters": 5,
        "sinkhorn_max_points": 64,
        "sinkhorn_kernel_clamp": 40.0,
        "sinkhorn_force_fp32": True,
        "sinkhorn_loss_clip": 5.0,
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1, 1],
        "use_adaptive_weights": False,
    }


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite {name}: {tensor}")


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cfg = _config()
    model = NoiseRobustCSWinModel(cfg).to(device)
    criterion = NoiseRobustLoss(cfg)
    disc_criterion = ComputeSinkhornDiscriminatorLoss(criterion)
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)

    rgb = torch.rand(2, 3, 16, 16, device=device)
    hsi = torch.rand(2, 31, 16, 16, device=device)

    start = time.perf_counter()
    model.train()
    if hasattr(model.generator, "set_iteration"):
        model.generator.set_iteration(0)

    optimizer_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_for_d = model.generator(rgb)
    real_pred = model.discriminator(rgb, hsi)
    fake_pred = model.discriminator(rgb, fake_for_d)
    disc_loss = disc_criterion(real_pred, fake_pred)
    _assert_finite("discriminator loss", disc_loss)
    disc_loss.backward()
    optimizer_d.step()

    optimizer_g.zero_grad(set_to_none=True)
    pred = model.generator(rgb)
    disc_real = model.discriminator(rgb, hsi).detach()
    disc_fake = model.discriminator(rgb, pred)
    gen_loss, _ = criterion(
        pred, hsi, disc_real=disc_real, disc_fake=disc_fake, current_iteration=1
    )
    _assert_finite("generator loss", gen_loss)
    gen_loss.backward()
    optimizer_g.step()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        val_pred = model(rgb[:1])
        val_metrics = compute_metrics(
            torch.clamp(val_pred, 0.0, 1.0),
            torch.clamp(hsi[:1], 0.0, 1.0),
            compute_all=True,
        )
        infer_1 = model(rgb[:1])
        infer_2 = model(rgb[:1])

    max_diff = (infer_1 - infer_2).abs().max().item()
    if max_diff > 1e-6:
        raise RuntimeError(f"Eval inference is nondeterministic: max_diff={max_diff}")

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(
        "smoke_run_ok "
        f"device={device} train_seconds={train_seconds:.3f} "
        f"peak_cuda_mb={peak_mb:.1f} "
        f"disc_loss={disc_loss.item():.6f} gen_loss={gen_loss.item():.6f} "
        f"val_psnr={val_metrics['psnr']:.3f} val_sam={val_metrics['sam']:.3f}"
    )


if __name__ == "__main__":
    main()
