#!/usr/bin/env python3
"""
Smoke-test training loop on synthetic data.

Runs a few iterations of the full training pipeline (forward, loss, backward,
optimizer step) on random tensors so that any shape/dtype/NaN bug surfaces
without requiring the ARAD-1K dataset.

Usage:
    python smoke_train.py [--steps 5] [--device cpu]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow imports from parent
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from model.mswr_net_v212 import create_mswr_tiny


def smoke_train(steps: int = 5, device_str: str = "cpu") -> dict:
    device = torch.device(device_str)
    print(f"[smoke_train] device={device}, steps={steps}")

    model = create_mswr_tiny(use_checkpoint=False).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    B, C_in, C_out, H, W = 2, 3, 31, 64, 64
    losses = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        rgb = torch.randn(B, C_in, H, W, device=device)
        hsi = torch.randn(B, C_out, H, W, device=device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                pred = model(rgb)
                loss = F.l1_loss(pred, hsi)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(rgb)
            loss = F.l1_loss(pred, hsi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        assert not torch.isnan(pred).any(), f"NaN in output at step {step}"
        assert pred.shape == hsi.shape, f"Shape mismatch at step {step}"

        print(f"  step {step}/{steps}  loss={loss_val:.6f}")

    elapsed = time.perf_counter() - t0
    avg_step = elapsed / steps * 1000

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    else:
        peak_mb = 0.0

    results = {
        "steps": steps,
        "device": device_str,
        "final_loss": losses[-1],
        "avg_step_ms": avg_step,
        "peak_vram_mb": peak_mb,
    }
    print(f"\n[smoke_train] PASSED  avg_step={avg_step:.1f}ms  peak_VRAM={peak_mb:.0f}MB")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    smoke_train(steps=args.steps, device_str=args.device)
