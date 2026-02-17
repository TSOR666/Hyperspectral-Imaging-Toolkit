#!/usr/bin/env python
"""
Minimal synthetic training smoke for HSIFusion and SHARP.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

from hsifusion_v252_complete import create_hsifusion_lightning_pro
from optimized_dataloader import MSTPlusPlusLoss
from sharp_v322_hardened import create_sharp_v32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def _run_hsifusion(device: torch.device, steps: int, batch_size: int, size: int) -> Dict[str, float]:
    if size < 64:
        raise ValueError("HSIFusion requires size >= 64 due min_input_size defaults.")

    model = create_hsifusion_lightning_pro(
        model_size="tiny",
        in_channels=3,
        out_channels=31,
        compile_mode=None,
        compile_model=False,
        force_compile=False,
    ).to(device)
    model.train()

    criterion = MSTPlusPlusLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    step_times: List[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(steps):
        rgb = torch.rand(batch_size, 3, size, size, device=device)
        hsi = torch.rand(batch_size, 31, size, size, device=device)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(rgb)
        if isinstance(out, tuple):
            out = out[0]
        loss = criterion(out, hsi)
        aux_loss = model.get_auxiliary_loss()
        if torch.is_tensor(aux_loss):
            loss = loss + aux_loss
        if not torch.isfinite(loss):
            raise RuntimeError("HSIFusion smoke loss became non-finite")
        loss.backward()
        optimizer.step()
        step_times.append(time.perf_counter() - t0)

    return {
        "loss": float(loss.detach().cpu().item()),
        "avg_step_s": float(np.mean(step_times)),
        "peak_vram_mb": _peak_vram_mb(),
    }


def _run_sharp(device: torch.device, steps: int, batch_size: int, size: int) -> Dict[str, float]:
    if size < 16:
        raise ValueError("SHARP smoke requires size >= 16.")

    model = create_sharp_v32(
        model_size="tiny",
        in_channels=3,
        out_channels=31,
        compile_model=False,
        verbose=False,
        sparse_max_tokens=2048,
        sparse_block_size=128,
        sparse_q_block_size=128,
        sparse_window_size=7,
        sparse_sparsity_ratio=0.5,
        sparse_k_cap=64,
    ).to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    step_times: List[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(steps):
        rgb = torch.rand(batch_size, 3, size, size, device=device)
        hsi = torch.rand(batch_size, 31, size, size, device=device)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(rgb)
        loss = model.compute_loss(out, hsi)
        if not torch.isfinite(loss):
            raise RuntimeError("SHARP smoke loss became non-finite")
        loss.backward()
        optimizer.step()
        step_times.append(time.perf_counter() - t0)

    return {
        "loss": float(loss.detach().cpu().item()),
        "avg_step_s": float(np.mean(step_times)),
        "peak_vram_mb": _peak_vram_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic smoke training for HSIFUSION&SHARP")
    parser.add_argument("--model", choices=["hsifusion", "sharp", "both"], default="both")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Running smoke_train on device={device}, model={args.model}, steps={args.steps}, size={args.size}")

    if args.model in {"hsifusion", "both"}:
        hsifusion_stats = _run_hsifusion(device, args.steps, args.batch_size, args.size)
        print(f"[HSIFusion] {hsifusion_stats}")

    if args.model in {"sharp", "both"}:
        sharp_stats = _run_sharp(device, args.steps, args.batch_size, args.size)
        print(f"[SHARP] {sharp_stats}")


if __name__ == "__main__":
    main()
