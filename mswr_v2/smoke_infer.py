#!/usr/bin/env python3
"""
Smoke-test inference on synthetic data.

Runs the model in eval mode on random tensors at various resolutions
to verify output shapes, absence of NaN/Inf, and determinism.

Usage:
    python smoke_infer.py [--device cpu]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from model.mswr_net_v212 import create_mswr_tiny


def smoke_infer(device_str: str = "cpu") -> dict:
    device = torch.device(device_str)
    print(f"[smoke_infer] device={device}")

    model = create_mswr_tiny().to(device)
    model.eval()

    resolutions = [(64, 64), (128, 128), (256, 256)]
    results = {}

    for h, w in resolutions:
        rgb = torch.randn(1, 3, h, w, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0

        assert out.shape == (1, 31, h, w), f"Shape mismatch for {h}x{w}: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in output for {h}x{w}"
        assert not torch.isinf(out).any(), f"Inf in output for {h}x{w}"

        # Determinism check
        with torch.no_grad():
            out2 = model(rgb)
        assert torch.allclose(out, out2, atol=1e-6), f"Non-deterministic output for {h}x{w}"

        key = f"{h}x{w}"
        results[key] = {"elapsed_ms": elapsed_ms, "peak_vram_mb": peak_mb}
        print(f"  {key}  time={elapsed_ms:.1f}ms  peak_VRAM={peak_mb:.0f}MB  shape={out.shape}")

    print(f"\n[smoke_infer] PASSED  all resolutions OK")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    smoke_infer(device_str=args.device)
