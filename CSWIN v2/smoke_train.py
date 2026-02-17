#!/usr/bin/env python3
"""Minimal synthetic training smoke run for CSWIN v2."""

from __future__ import annotations

import sys
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


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
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
    }

    model = NoiseRobustCSWinModel(config).to(device)
    criterion = NoiseRobustLoss(config)
    disc_criterion = ComputeSinkhornDiscriminatorLoss(criterion)

    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)

    rgb = torch.rand(2, 3, 8, 8, device=device)
    hsi = torch.rand(2, 31, 8, 8, device=device)

    model.train()

    optimizer_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_hsi_detached = model.generator(rgb)
    real_pred = model.discriminator(rgb, hsi)
    fake_pred = model.discriminator(rgb, fake_hsi_detached)
    disc_loss = disc_criterion(real_pred, fake_pred)
    if not torch.isfinite(disc_loss):
        raise RuntimeError(f"Non-finite discriminator loss: {disc_loss}")
    disc_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
    optimizer_d.step()

    optimizer_g.zero_grad(set_to_none=True)
    pred_hsi = model.generator(rgb)
    disc_real = model.discriminator(rgb, hsi).detach()
    disc_fake = model.discriminator(rgb, pred_hsi)
    gen_loss, _ = criterion(
        pred_hsi, hsi, disc_real=disc_real, disc_fake=disc_fake, current_iteration=1
    )
    if not torch.isfinite(gen_loss):
        raise RuntimeError(f"Non-finite generator loss: {gen_loss}")
    gen_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
    optimizer_g.step()

    print(
        f"smoke_train_ok device={device} "
        f"disc_loss={disc_loss.item():.6f} gen_loss={gen_loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
