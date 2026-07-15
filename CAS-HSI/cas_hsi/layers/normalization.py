"""Bias-free channel normalization (specification section 4.3).

Normalizes over the channel dimension independently at every spatial location.
The bias-free form avoids re-centering, which is standard for image restoration
backbones (Restormer / NAFNet lineage).

BatchNorm is deliberately absent from the whole package: batch statistics are
wrong for patch-trained / full-image-evaluated restoration, and they break the
arbitrary-size and export requirements.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BiasFreeLayerNorm2d", "RMSNorm2d"]


class BiasFreeLayerNorm2d(nn.Module):
    """LayerNorm over channels, per pixel, without mean subtraction or bias.

    ``LN(x_p) = x_p / sqrt(Var(x_p) + eps) * w``

    Statistics are computed in fp32 even under autocast (spec 9.5: "Normalization
    statistics: FP32"), then cast back to the input dtype.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.var(dim=1, keepdim=True, unbiased=False)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_fp32 * self.weight.float().view(1, -1, 1, 1)).to(dtype)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, eps={self.eps}"


class RMSNorm2d(nn.Module):
    """Root-mean-square norm over channels (spec 4.3 "Alternative: RMSNorm2d").

    Differs from :class:`BiasFreeLayerNorm2d` in using the second moment rather
    than the variance, i.e. it does not subtract the mean when measuring scale.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        mean_square = x_fp32.pow(2).mean(dim=1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(mean_square + self.eps)
        return (x_fp32 * self.weight.float().view(1, -1, 1, 1)).to(dtype)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, eps={self.eps}"


def build_norm(name: str, channels: int, eps: float = 1e-6) -> nn.Module:
    """Factory for the two permitted normalizations. BatchNorm is not offered."""
    key = str(name).strip().lower()
    if key in {"layernorm", "layer", "biasfree", "biasfree_layernorm"}:
        return BiasFreeLayerNorm2d(channels, eps=eps)
    if key in {"rms", "rmsnorm", "rmsnorm2d"}:
        return RMSNorm2d(channels, eps=eps)
    raise ValueError(
        f"Unknown norm {name!r}; expected 'layernorm' or 'rmsnorm'. "
        "BatchNorm is not permitted in the restoration backbone (spec 4.3)."
    )
