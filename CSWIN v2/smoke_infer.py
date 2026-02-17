#!/usr/bin/env python3
"""Minimal synthetic inference smoke run for CSWIN v2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hsi_model.models import NoiseRobustCSWinModel  # noqa: E402
from hsi_model.utils.metrics import compute_metrics  # noqa: E402


def main() -> None:
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1, 1],
    }

    model = NoiseRobustCSWinModel(config).to(device).eval()
    rgb = torch.rand(1, 3, 16, 16, device=device)

    with torch.no_grad():
        pred_1 = model(rgb)
        pred_2 = model(rgb)

    if pred_1.shape != (1, 31, 16, 16):
        raise RuntimeError(f"Unexpected output shape: {pred_1.shape}")

    max_diff = torch.max(torch.abs(pred_1 - pred_2)).item()
    if max_diff > 1e-6:
        raise RuntimeError(f"Inference is non-deterministic in eval mode (max_diff={max_diff})")

    target = torch.clamp(pred_1 * 0.97 + 0.01, 0.0, 1.0)
    metrics = compute_metrics(torch.clamp(pred_1, 0.0, 1.0), target, compute_all=False)

    if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
        raise RuntimeError(f"Non-finite metrics detected: {metrics}")

    print(f"smoke_infer_ok device={device} max_diff={max_diff:.2e} metrics={metrics}")


if __name__ == "__main__":
    main()
