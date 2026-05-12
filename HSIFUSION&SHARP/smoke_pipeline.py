#!/usr/bin/env python
"""
Synthetic pipeline smoke for HSIFusion and SHARP.
Runs one train step, one validation step, and one inference pass.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from hsifusion_v252_complete import create_hsifusion_lightning_pro
from optimized_dataloader import MSTPlusPlusLoss
from sharp_inference import SHARPInference
from sharp_v322_hardened import SHARPv32Trainer, create_sharp_v32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mrae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    denom = torch.clamp_min(torch.abs(target), eps)
    return float(torch.mean(torch.abs(pred - target) / denom).item())


def _psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-8)
    return float((10.0 * torch.log10(torch.tensor(data_range ** 2, device=mse.device) / mse)).item())


def _run_hsifusion(device: torch.device, batch_size: int, size: int) -> Dict[str, float]:
    if size < 64:
        raise ValueError("HSIFusion requires size >= 64 due to min_input_size defaults.")

    model = create_hsifusion_lightning_pro(
        model_size="tiny",
        in_channels=3,
        out_channels=31,
        compile_mode=None,
        compile_model=False,
        force_compile=False,
    ).to(device)
    criterion = MSTPlusPlusLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    rgb = torch.rand(batch_size, 3, size, size, device=device)
    hsi = torch.rand(batch_size, 31, size, size, device=device)

    train_t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(rgb)
    if isinstance(pred, tuple):
        pred = pred[0]
    train_loss = criterion(pred, hsi)
    aux_loss = model.get_auxiliary_loss()
    if torch.is_tensor(aux_loss):
        train_loss = train_loss + aux_loss
    train_loss.backward()
    optimizer.step()
    train_elapsed = time.perf_counter() - train_t0

    val_rgb = torch.rand(1, 3, size, size, device=device)
    val_hsi = torch.rand(1, 31, size, size, device=device)
    val_t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        val_pred = model(val_rgb)
        if isinstance(val_pred, tuple):
            val_pred = val_pred[0]
        val_loss = criterion(val_pred, val_hsi)
    val_elapsed = time.perf_counter() - val_t0

    infer_t0 = time.perf_counter()
    with torch.no_grad():
        infer_pred = model(val_rgb)
        if isinstance(infer_pred, tuple):
            infer_pred = infer_pred[0]
    infer_elapsed = time.perf_counter() - infer_t0

    return {
        "train_loss": float(train_loss.detach().cpu().item()),
        "val_loss": float(val_loss.detach().cpu().item()),
        "val_mrae": _mrae(val_pred, val_hsi),
        "val_psnr": _psnr(val_pred, val_hsi, data_range=1.0),
        "train_s": float(train_elapsed),
        "val_s": float(val_elapsed),
        "infer_s": float(infer_elapsed),
        "finite": float(torch.isfinite(infer_pred).all().item()),
    }


def _run_sharp(device: torch.device, batch_size: int, size: int) -> Dict[str, float]:
    if size < 16:
        raise ValueError("SHARP requires size >= 16.")

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
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    rgb = torch.rand(batch_size, 3, size, size, device=device)
    hsi = torch.rand(batch_size, 31, size, size, device=device)

    train_t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(rgb)
    train_loss = model.compute_loss(pred, hsi)
    train_loss.backward()
    optimizer.step()
    train_elapsed = time.perf_counter() - train_t0

    trainer = SHARPv32Trainer(model=model, total_steps=4, use_amp=False)
    val_loader = DataLoader(
        TensorDataset(torch.rand(1, 3, size, size), torch.rand(1, 31, size, size)),
        batch_size=1,
    )
    val_t0 = time.perf_counter()
    val_metrics = trainer.evaluate(val_loader, psnr_max=1.0)
    val_elapsed = time.perf_counter() - val_t0

    work_dir = Path(".tmp_smoke_sharp_pipeline")
    work_dir.mkdir(exist_ok=True)
    ckpt_path = work_dir / "smoke_ckpt.pth"
    cfg = SimpleNamespace(
        model_size="tiny",
        in_channels=3,
        out_channels=31,
        sparse_sparsity_ratio=0.5,
        rbf_centers_per_head=32,
        sparse_k_cap=64,
        sparse_block_size=128,
        sparse_q_block_size=128,
        sparse_window_size=7,
        sparse_max_tokens=2048,
        key_rbf_mode="mean",
        sparsemax_pad_value=None,
        ema_update_every=1,
    )
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    rgb_np = (np.random.rand(size, size, 3) * 255.0).astype(np.uint8)
    input_path = work_dir / "smoke_rgb.png"
    output_path = work_dir / "smoke_hsi.npy"
    Image.fromarray(rgb_np).save(input_path)

    infer_t0 = time.perf_counter()
    runner = SHARPInference(str(ckpt_path), device=str(device))
    infer_hsi = runner.process_image_file(
        str(input_path),
        output_path=str(output_path),
        patch_size=max(8, size // 2),
    )
    infer_elapsed = time.perf_counter() - infer_t0

    return {
        "train_loss": float(train_loss.detach().cpu().item()),
        "val_loss": float(val_metrics["loss"]),
        "val_mrae": float(val_metrics["mrae"]),
        "val_psnr": float(val_metrics["psnr"]),
        "train_s": float(train_elapsed),
        "val_s": float(val_elapsed),
        "infer_s": float(infer_elapsed),
        "finite": float(np.isfinite(infer_hsi).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic pipeline smoke for HSIFUSION&SHARP")
    parser.add_argument("--model", choices=["hsifusion", "sharp", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(
        f"Running smoke_pipeline on device={device}, model={args.model}, "
        f"batch_size={args.batch_size}, size={args.size}"
    )

    if args.model in {"hsifusion", "both"}:
        hsifusion_stats = _run_hsifusion(device, args.batch_size, args.size)
        print(f"[HSIFusion] {hsifusion_stats}")

    if args.model in {"sharp", "both"}:
        sharp_stats = _run_sharp(device, args.batch_size, max(16, args.size))
        print(f"[SHARP] {sharp_stats}")


if __name__ == "__main__":
    main()
