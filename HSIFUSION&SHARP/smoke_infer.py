#!/usr/bin/env python
"""
Minimal synthetic inference smoke for HSIFusion and SHARP.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from PIL import Image

from hsifusion_v252_complete import create_hsifusion_lightning_pro
from sharp_inference import SHARPInference
from sharp_v322_hardened import create_sharp_v32


def _run_hsifusion(device: torch.device, size: int) -> Dict[str, float]:
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
    model.eval()

    x = torch.rand(1, 3, size, size, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
    elapsed = time.perf_counter() - t0

    if out.shape != (1, 31, size, size):
        raise RuntimeError(f"Unexpected HSIFusion output shape: {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise RuntimeError("HSIFusion inference output contains non-finite values")

    return {"infer_s": float(elapsed)}


def _run_sharp(device: torch.device, size: int) -> Dict[str, float]:
    work_dir = Path(".tmp_smoke_sharp")
    work_dir.mkdir(exist_ok=True)

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
    )
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

    rgb = (np.random.rand(size, size, 3) * 255.0).astype(np.uint8)
    input_path = work_dir / "smoke_rgb.png"
    output_path = work_dir / "smoke_hsi.npy"
    Image.fromarray(rgb).save(input_path)

    runner = SHARPInference(str(ckpt_path), device=str(device))
    t0 = time.perf_counter()
    hsi = runner.process_image_file(
        str(input_path),
        output_path=str(output_path),
        patch_size=max(8, size // 2),
    )
    elapsed = time.perf_counter() - t0

    if hsi.shape != (31, size, size):
        raise RuntimeError(f"Unexpected SHARP output shape: {hsi.shape}")
    if not np.isfinite(hsi).all():
        raise RuntimeError("SHARP inference output contains non-finite values")
    if not output_path.exists():
        raise RuntimeError("SHARP smoke output file was not written")

    return {"infer_s": float(elapsed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic smoke inference for HSIFUSION&SHARP")
    parser.add_argument("--model", choices=["hsifusion", "sharp", "both"], default="both")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running smoke_infer on device={device}, model={args.model}, size={args.size}")

    if args.model in {"hsifusion", "both"}:
        hsifusion_stats = _run_hsifusion(device, args.size)
        print(f"[HSIFusion] {hsifusion_stats}")

    if args.model in {"sharp", "both"}:
        sharp_stats = _run_sharp(device, args.size)
        print(f"[SHARP] {sharp_stats}")


if __name__ == "__main__":
    main()
