"""Shared numerical-stability helpers for HSIFusion and SHARP training."""

from __future__ import annotations

import contextlib
import math
from typing import ContextManager, Optional

import torch


AMP_MODES = ("auto", "bf16", "fp16", "off")


def resolve_amp_dtype(mode: str, device: torch.device) -> Optional[torch.dtype]:
    """Resolve an AMP policy, preferring BF16 because it has FP32's exponent range."""
    mode = str(mode).lower()
    if mode not in AMP_MODES:
        raise ValueError(f"amp must be one of {AMP_MODES}, got {mode!r}")
    if mode == "off" or device.type != "cuda":
        return None
    bf16_supported = bool(
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    if mode == "auto":
        return torch.bfloat16 if bf16_supported else torch.float16
    if mode == "bf16":
        if not bf16_supported:
            raise RuntimeError(
                "BF16 AMP was requested, but this CUDA device does not support it"
            )
        return torch.bfloat16
    return torch.float16


def autocast_context(
    device: torch.device, dtype: Optional[torch.dtype]
) -> ContextManager[None]:
    """Return an autocast context compatible with both old and new PyTorch APIs."""
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=dtype)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def make_grad_scaler(
    device: torch.device,
    dtype: Optional[torch.dtype],
    init_scale: float = 1024.0,
):
    """Create a scaler only for FP16; BF16 does not need or benefit from scaling."""
    if not math.isfinite(init_scale) or init_scale <= 0:
        raise ValueError("fp16_init_scale must be a finite positive number")
    enabled = device.type == "cuda" and dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(
                "cuda", enabled=enabled, init_scale=float(init_scale)
            )
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled, init_scale=float(init_scale))
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=float(init_scale))


def annealed_mrae_epsilon(
    start: float,
    end: float,
    step: int,
    anneal_steps: int,
) -> float:
    """Log-linearly anneal a stable training floor without changing validation MRAE."""
    start = float(start)
    end = float(end)
    if not math.isfinite(start) or start <= 0:
        raise ValueError("train_mrae_eps_start must be a finite positive number")
    if not math.isfinite(end) or end <= 0:
        raise ValueError("train_mrae_eps_end must be a finite positive number")
    if anneal_steps < 0:
        raise ValueError("train_mrae_eps_anneal_steps must be >= 0")
    if anneal_steps == 0:
        return end
    progress = min(max(float(step), 0.0), float(anneal_steps)) / float(anneal_steps)
    return float(math.exp(math.log(start) + progress * (math.log(end) - math.log(start))))
